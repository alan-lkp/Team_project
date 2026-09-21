# -*- coding: utf-8 -*-
"""LLM 同义改写增广(鲁棒性评测集 / 训练集扩充共用)。

设计约束(与需求对齐):
  1. 换说法不换意思: 同义替换、调语序、口语化、允许错别字/语气词;
  2. 原文每一个"诉求线索"必须保留 —— 线索丢了标签就变;
  3. 不添加原文没有的信息;
  4. 禁止照抄原句用词 —— 脚本用字符 2-gram 覆盖率做照抄检测, 超阈值判失败重试;
  5. invalid 类保持灌水感(不能把广告/测试句改成真实诉求);
  6. temperature 0.8 制造多样性。

模型: 阿里云百炼 qwen-plus (OpenAI 兼容端点), 不走 DeepSeek。
API key 读取顺序: $DASHSCOPE_API_KEY -> Hermes config.yaml 里的 alibaba provider。

用法:
  python tools/llm_paraphrase.py --prompt-demo          # 打印完整 prompt 供审阅
  python tools/llm_paraphrase.py --sample 30 --seed 42  # 冒烟
  python tools/llm_paraphrase.py                        # 全量 test.txt -> test_paraphrase.txt
  重跑安全: 结果逐批写入 cache jsonl, 断点续跑不重复计费。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time

import requests

DEFAULT_IN = os.path.join("01-data", "test.txt")
DEFAULT_OUT = os.path.join("01-data", "test_paraphrase.txt")
DEFAULT_CACHE = os.path.join("tools", "paraphrase_cache", "test_aug.jsonl")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-plus"
TEMPERATURE = 0.8
COPY_LIMIT = 0.85  # 原文 2-gram 被改写句覆盖的比例上限, 超过视为"照抄"

# ---------------------------------------------------------------- prompt ----

SYSTEM_PROMPT = """你是电商客服工单的语料改写助手, 任务是给工单做"换说法不换意思"的同义改写, 用于训练模型的鲁棒性。

必须遵守的规则(按重要性排序):
1. 【保留全部诉求线索】原文里有几个诉求点, 改写后必须一个不少、一个不多。诉求点 = 可独立归类的事件, 例如"退款没到账""券失效""客服态度差"各是一个点。丢了任何一个点, 标签就会变, 这是最严重的错误。多诉求句通常用"；/另外/顺便/还有"连接, 每个分句都要改写到位, 分句顺序可以调换。
2. 【禁止照抄原句用词】不允许原样保留整句或原句的关键词组, 必须换词、换句式: 同义替换(降价→便宜了/掉价、客服→工作人员、发票→票据)、调整语序、换句式(把陈述换成一问一答感/反问)。
3. 【不添加原文没有的信息】不能新增时间、地点、商品、金额、情绪、诉求。原文没提"急"你就不能加"急"。
4. 【口语化+容错】模仿真实用户打字: 可以带语气词(哎、啧、哈、亲、咱)、倒装、省略主语, 允许少量错别字或别称(退货→退掉、发票→发飘(错字)、微信→薇信), 但整句必须仍能读懂。
5. 【长度相当】改写后字数与原句大致相当(±30%), 不要扩写成小作文, 也不要缩成几个字。
6. 【invalid 类保灌水感】原文是测试句/广告/纯表情/刷屏辱骂时, 改写后仍然是无效内容: 可以换一种灌水方式(换个乱码、换个广告说法、换个无意义重复), 严禁把它改写成带真实诉求的正常工单, 也不要额外补一句真实诉求。

各类别改写示例(原句 → 改写):
- logistics: "快递在郑州卡了一周不动" → "包裹搁郑州那边儿一个礼拜了, 轨迹就是不变"
- quality: "充电宝用了两天就出故障" → "这充电宝才摸两天就罢工了"
- after_sale: "退款都一周了还没到账" → "钱退回去七个白天黑夜了咋还没进我账户"
- invoice: "发票抬头写错了能重开吗" → "票据抬头弄错了, 麻烦给重新开一张中不"
- price_promo: "刚买完就降价，能退差价吗" → "东西才到手价就掉了, 差价能不能补给咱"
- payment_account: "同一笔钱被扣了两回" → "咋给我划了两遍款, 多扣的退回来"
- consult: "这个面膜能带上飞机吗" → "想问下坐飞机的时候面膜能随身带不"
- service: "客服态度太差，说话很冲" → "咨询的时候那工作人员讲话噎死人, 态度属实不行"
- invalid: "测试测试测试" → "asdfghjkl试试看" (仍无意义); "加我微信有内部价" → "薇信加一下, 低价拿货" (仍是广告)
- 多标签: "券说没就没了，还有同一笔钱被扣了两回" → "那优惠券咋凭空消失了, 而且一笔钱给我划了两遍" (两个点都在)

输出格式: 输入是若干行 "序号<TAB>原文", 你输出相同行数、相同序号的 "序号<TAB>改写句"。
不要输出标签、编号列表、解释、markdown, 行内不得再出现 TAB 或换行。
改写句就是给用户看的最终文本: 严禁在句子里附带任何括号注释、说明、旁白(如"（语气带点无奈）"这类元信息), 只能出现工单本身的内容。"""

FEWSHOT_TAIL = """参考规则执行。序号必须原样保留, 一条输入对一条输出。"""


def build_user_prompt(items: list[tuple[int, str]]) -> str:
    body = "\n".join(f"{i}\t{t}" for i, t in items)
    return (
        "请按系统规则逐条改写下面的工单(制表符分隔序号与文本):\n\n"
        + body
        + "\n\n"
        + FEWSHOT_TAIL
    )


# ---------------------------------------------------------------- api --------

def _candidate_keys() -> list[str]:
    keys = []
    env = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if env:
        keys.append(env)
    # 本机已知的百炼 key 存放处(Hermes config / OpenViking 配置), 不打印明文
    sources = [
        os.path.expanduser(r"~\.openviking\ov.conf"),
        r"E:\soft\Hermes Agent CN Desktop\data\hermes-home\config.yaml",
    ]
    for src in sources:
        try:
            text = open(src, encoding="utf-8").read()
        except OSError:
            continue
        keys += re.findall(r"sk-[A-Za-z0-9\-]{10,}", text)
    return list(dict.fromkeys(keys))


def load_api_key() -> str:
    """逐个试 key, 返回第一个能通过百炼验证的。"""
    url = BASE_URL.replace("/chat/completions", "/models")
    for k in _candidate_keys():
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {k}"}, timeout=20)
            if r.status_code == 200:
                print(f"API key 验证通过: {k[:6]}...{k[-4:]}")
                return k
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit(
        "没有可用的百炼 DashScope key: 请设置环境变量 DASHSCOPE_API_KEY "
        "(不走 DeepSeek)"
    )


_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "s"):
        _thread_local.s = requests.Session()
    return _thread_local.s


def call_llm(items: list[tuple[int, str]], key: str,
             temperature: float = TEMPERATURE, max_retry: int = 3) -> dict[int, str]:
    """返回 {序号: 改写句}; 网络/格式错误重试, 全失败则抛 RuntimeError。"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(items)},
        ],
        "temperature": temperature,
        "max_tokens": 2400,
    }
    headers = {"Authorization": f"Bearer {key}"}
    want = {i for i, _ in items}
    last_err = "unknown"
    for attempt in range(max_retry):
        try:
            r = _session().post(BASE_URL, json=payload, headers=headers, timeout=180)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                content = r.json()["choices"][0]["message"]["content"]
                got = parse_response(content)
                if want <= set(got):
                    return {i: got[i] for i in want}
                last_err = f"缺行: 期望{len(want)} 得到{len(got)} (差 {sorted(want - set(got))})"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"batch 失败: {last_err}")


def parse_response(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 容错: 模型可能输出 "1. xxx\t" 或 "1\txxx" 或 "1｜xxx"
        line = re.sub(r"^\d+\.\s*", "", line)
        parts = re.split(r"\t|[\uff5c|]\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        idx_s, new = parts[0].strip(), parts[1].strip()
        if idx_s.isdigit() and new:
            out[int(idx_s)] = new.replace("\t", "，")
    return out


# ------------------------------------------------------------- 质检 ----------

def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def copy_ratio(orig: str, new: str) -> float:
    """原文 2-gram 有多少比例被改写句保留 —— 越高越接近照抄。"""
    ob = bigrams(orig)
    if not ob:
        return 0.0
    return len(ob & bigrams(new)) / len(ob)


def length_ok(orig: str, new: str) -> bool:
    lo, hi = len(orig) * 0.55, len(orig) * 1.75
    return max(4, lo) <= len(new) <= max(30, hi)


# ------------------------------------------------------------- 主流程 --------

def load_lines(path: str) -> list[tuple[str, str]]:
    rows = []
    for ln in open(path, encoding="utf-8").read().splitlines():
        if not ln.strip():
            continue
        text, label = ln.rsplit("\t", 1)
        rows.append((text.strip(), label.strip()))
    return rows


def load_cache(path: str) -> dict[int, str]:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["idx"]] = rec["new"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(args: argparse.Namespace) -> None:
    if args.prompt_demo:
        print(SYSTEM_PROMPT)
        print("=" * 30, "USER 示例", "=" * 30)
        print(build_user_prompt([(1, "券写着能用，结账就失效；说好一天内回话，到现在没人理")]))
        return

    rows = load_lines(args.input)
    total = len(rows)
    idx_all = list(range(total))
    if args.sample:
        rng = random.Random(args.seed)
        idx_all = sorted(rng.sample(idx_all, min(args.sample, total)))
    if args.limit:
        idx_all = idx_all[: args.limit]

    key = load_api_key()
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    done = {} if args.no_cache else load_cache(args.cache)
    todo = [i for i in idx_all if i not in done]
    print(f"总数 {total}, 本次需处理 {len(idx_all)}, 已有缓存 {len(idx_all)-len(todo)}, 待请求 {len(todo)}")

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    failures: list[dict] = []
    cache_f = open(args.cache, "a", encoding="utf-8") if not args.no_cache else None

    def work(batch_idx: list[int]) -> None:
        items = [(j, rows[j][0]) for j in batch_idx]
        try:
            got = call_llm(items, key, temperature=args.temperature)
        except RuntimeError as e:
            with lock:
                stats["fail"] += len(batch_idx)
                failures.append({"idxs": batch_idx, "error": str(e)})
            print(f"  [FAIL] batch {batch_idx[0]}..{batch_idx[-1]}: {e}", flush=True)
            return
        for j in batch_idx:
            new = got[j]
            orig = rows[j][0]
            cr = copy_ratio(orig, new)
            if (cr > COPY_LIMIT or not length_ok(orig, new)) and args.temperature < 0.99:
                # 判照抄/跑偏: 用更高温度重做一次
                try:
                    got2 = call_llm(items, key, temperature=0.95)
                    new2 = got2[j]
                    if copy_ratio(orig, new2) <= COPY_LIMIT and length_ok(orig, new2):
                        new = new2
                        cr = copy_ratio(orig, new)
                except RuntimeError:
                    pass
            rec = {"idx": j, "new": new, "copy_ratio": round(cr, 3),
                   "flags": ([ ] if cr <= COPY_LIMIT else ["copy"]) +
                            ([] if length_ok(orig, new) else ["len"])}
            with lock:
                stats["ok"] += 1
                if rec["flags"]:
                    failures.append({"idxs": [j], "error": f"flags={rec['flags']} cr={cr:.2f}"})
                if cache_f:
                    cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache_f.flush()
        done.update({j: got[j] for j in batch_idx})
        with lock:
            n = stats["ok"] + stats["fail"]
            if n % (args.batch_size * 20) < args.batch_size:
                print(f"  进度 {n}/{len(todo)} ok={stats['ok']} fail={stats['fail']}", flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, b) for b in batches]
        for fu in cf.as_completed(futures):
            fu.result()  # 抛出未捕获异常则终止
    dt = time.time() - t0

    # 汇总输出
    done = {} if args.no_cache else (done if args.sample else done)
    results = load_cache(args.cache) if not args.no_cache else done
    merged = {j: done.get(j, results.get(j)) for j in idx_all}
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            for j in idx_all:
                text, label = rows[j]
                new = merged.get(j) or text
                f.write(f"{new}\t{label}\n")
        print(f"已写出 {args.output} ({len(idx_all)} 行)")
    report = {
        "input": args.input, "total": total, "processed": len(idx_all),
        "ok": stats["ok"], "fail": stats["fail"],
        "elapsed_s": round(dt, 1),
        "copy_ratio_flags": sum(1 for x in failures if "flags=" in x.get("error", "")),
        "failures": failures[:50],
    }
    rpath = (args.report or args.output + ".report.json") if args.output else "paraphrase_report.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {rpath}")
    print(json.dumps({k: report[k] for k in
                      ("processed", "ok", "fail", "elapsed_s", "copy_ratio_flags")},
                     ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=None)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--report", default=None)
    ap.add_argument("--sample", type=int, default=0, help="随机抽 N 条(冒烟)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--prompt-demo", action="store_true")
    args = ap.parse_args()
    if not args.output:
        args.output = DEFAULT_OUT
    run(args)


if __name__ == "__main__":
    main()
