# -*- coding: utf-8 -*-
"""对 test_paraphrase.txt 做终检修补:
  A. 剥离模型偶尔漏进来的括号旁白(如"（纯乱码感保留）");
  B. 与原文完全相同的行 -> 重改写;
  C. 改写集内部重复(同一条改写出现多次) -> 除首例外强制差异化重改写。
重改写结果回写 cache, 保持断点续跑语义。
"""
import json, re, sys, collections
sys.path.insert(0, 'tools')
from llm_paraphrase import call_llm, load_api_key, copy_ratio, bigrams

ORIG_F, NEW_F = '01-data/test.txt', '01-data/test_paraphrase.txt'
CACHE_F = 'tools/paraphrase_cache/test_aug.jsonl'

orig = [l for l in open(ORIG_F, encoding='utf-8').read().splitlines() if l.strip()]
new  = [l for l in open(NEW_F, encoding='utf-8').read().splitlines() if l.strip()]
assert len(orig) == len(new)
rows = [(l.rsplit('\t',1)[0], l.rsplit('\t',1)[1]) for l in new]

# ---- A. 剥离旁白括号 -------------------------------------------------------
NOISE = re.compile(r'[（(][^（）()]{0,24}(语气|说明|旁白|注[:：]|纯乱码|纯笑声|纯数字|测试用|手动狗头|无意义|无诉求)[^（）()]{0,24}[）)]')
fixedA = 0
for i,(t,lab) in enumerate(rows):
    t2 = NOISE.sub('', t).strip('，,。；;、 ')
    if t2 and t2 != t and len(t2) >= 2:
        rows[i] = (t2, lab); fixedA += 1
print("A 剥离旁白:", fixedA)

# ---- B/C. 确定需要重改写的行 ----------------------------------------------
otexts = [l.rsplit('\t',1)[0] for l in orig]
need = set()
identical = [i for i,(t,_) in enumerate(rows) if t == otexts[i]]
need.update(identical)
cnt = collections.Counter(t for t,_ in rows)
dup_texts = {t for t,c in cnt.items() if c > 1}
seen = set()
for i,(t,_) in enumerate(rows):
    if t in dup_texts:
        if t in seen:
            need.add(i)          # 保留首现, 其余差异化
        seen.add(t)
need = sorted(need)
print("B/C 待重改写:", len(need), "(与原文同:", len(identical), ", 重复副本:", len(need)-len(identical), ")")

# ---- 重改写: 单条并发, 差异化时把"撞车句"喂给模型禁抄 ----------------------
key = load_api_key()
cache = [json.loads(l) for l in open(CACHE_F, encoding='utf-8')]
cache_map = {r['idx']: r['new'] for r in cache}
text2idx = collections.defaultdict(list)
for i,(t,_) in enumerate(rows):
    text2idx[t].append(i)

EXTRA = {i: "该行改写与集合内其他行撞车了, 必须换一种明显不同的说法(灌水类: 换另一串乱码/另一种无意义重复/另一种广告措辞)"
         for i in need if len(text2idx[rows[i][0]]) > 1}

results = {}
def attempt(i, tag):
    extra = EXTRA.get(i, "")
    item = (1, orig[i].rsplit('\t',1)[0] + (f"  [附加指令: {extra}]" if extra else ""))
    got = call_llm([item], key, temperature=0.95)
    cand = got[1]
    cand = NOISE.sub('', cand).strip()
    cr = copy_ratio(otexts[i], cand)
    clash = cand in text2idx and text2idx[cand] != [i]
    if cand and cr <= 0.85 and not clash and len(cand) >= 2:
        results[i] = cand
        print(f"  {i} [{tag}] ok cr={cr:.2f}: {cand}")
        return True
    print(f"  {i} [{tag}] 不合格 cr={cr:.2f} clash={clash}: {cand}")
    return False

fails = []
for i in need:
    ok = False
    for tag in ("r1","r2","r3"):
        try:
            ok = attempt(i, tag)
        except RuntimeError as e:
            print(f"  {i} [{tag}] API失败: {e}")
        if ok: break
    if not ok:
        fails.append(i)

for i, t in results.items():
    rows[i] = (t, rows[i][1])
    cache_map[i] = t

with open(CACHE_F, 'w', encoding='utf-8') as f:
    for i,t in cache_map.items():
        f.write(json.dumps({"idx": i, "new": t}, ensure_ascii=False) + "\n")
with open(NEW_F, 'w', encoding='utf-8', newline='\n') as f:
    for t, lab in rows:
        f.write(f"{t}\t{lab}\n")
print("重改写成功:", len(results), "仍失败(保留原改写):", fails)

# ---- 复检 ------------------------------------------------------------------
final = [l.rsplit('\t',1)[0] for l in open(NEW_F, encoding='utf-8').read().splitlines()]
assert all(final[i] == rows[i][0] for i in range(len(rows)))
dup2 = sum(c-1 for c in collections.Counter(final).values() if c>1)
noise2 = sum(1 for t in final if NOISE.search(t))
ident2 = sum(1 for i,t in enumerate(final) if t == otexts[i])
def big(s):
    s = re.sub(r'\s+','',s); return {s[i:i+2] for i in range(len(s)-1)} if len(s)>1 else {s}
crs = [len(big(otexts[i])&big(final[i]))/len(big(otexts[i])) for i in range(len(final)) if big(otexts[i])]
over = sum(1 for c in crs if c > 0.85)
print(f"复检: 内部重复={dup2} 旁白={noise2} 与原文同={ident2} 照抄>0.85={over}")
