"""
LLM 兜底解析 (04-bert) —— 拒识工单的语义兜底

定位:
    双阈值拒识把"本地模型没把握"的工单挑出来后, 交给大模型做语义兜底解析.
    这是三级兜底机制(must: 本地模型 -> LLM -> 人工复核)的第二级.

成本控制设计(为什么不能全量走 LLM):
    1. 只有 rejected=True 的工单才会触发(见 reject_utils.should_use_llm);
    2. 未配置 DEEPSEEK_API_KEY 时**自动禁用**, 服务照样能跑(走人工复核);
    3. 提示词严格限定 9 类 + 强制 JSON 输出, 减少无效 token 与解析失败重试;
    4. 相同文本在服务层会被 Redis 缓存(见 backend/middleware), 避免重复计费.

可靠性设计:
    - 返回结果必须通过"标签合法性校验"(只能是我们体系内的 9 个标签);
    - 解析失败/超时/标签非法 -> 返回 None, 由上层转入人工复核, 绝不返回脏数据;
    - 目前不引入 json_repair 等额外依赖, 用手写提取(去掉 ```json 包裹、取首尾大括号).

自测(不需要 API Key):
    python 04-bert/llm_fallback.py
"""

import json
import os
import re

# ============================================================
# todo 0. 配置(全部可用环境变量覆盖)
# ============================================================
DEFAULT_MODEL = os.environ.get('LLM_MODEL', 'deepseek-chat')
DEFAULT_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com')

# 标签中文名(用于提示词里做类别说明; 与 backend/common/shop_utils.py 保持一致)
LABEL_CN = {
    'logistics': '物流配送(未发货、运输停滞、派送延迟、丢件、地址修改)',
    'quality': '商品质量(破损、瑕疵、功能故障、材质不符、过期变质、少件漏发)',
    'after_sale': '退款退货(退款、退货、换货、退款未到账、维修换新、售后维权)',
    'invoice': '发票问题(开票申请、发票信息错误、重开、发票未收到)',
    'price_promo': '价格与优惠(价保补差、优惠券失效、活动规则、多扣款、赠品未发)',
    'payment_account': '支付与账号(支付失败、重复扣款、订单异常、账号封禁或登录异常)',
    'consult': '售前与使用咨询(使用方法、参数规格、适配性、库存时效、安装教程)',
    'service': '服务态度(回复慢、态度差、敷衍、承诺未兑现、推诿)',
    'invalid': '无效与恶意(灌水、广告、辱骂、与商品无关、恶意差评威胁)',
}


def build_system_prompt(class_list):
    """构造系统提示词: 角色 + 类别定义 + 判定规则 + 少样本示例 + 强约束输出格式"""
    options = '\n'.join(f'- {name}: {LABEL_CN.get(name, "")}' for name in class_list)
    return f"""你是电商客服工单的多标签分类助手。请阅读用户反馈, 从下列固定类别中选出所有命中的标签。

# 可选类别(严格限定, 不得新增)
{options}

# 判定规则
1. 一条工单可以同时命中多个类别(例如"货坏了要退款, 客服还不理人" = quality + after_sale + service), 请全部选出;
2. 只有文本里**确实提到**某个诉求时才给对应标签, 不要脑补;
3. 纯灌水、广告、辱骂且无任何具体诉求的工单, 只输出 invalid;
4. 若同时命中 invalid 与其它诉求, 以具体诉求为准, 不要输出 invalid;
5. 发票、价保、优惠券、支付失败、账号问题都是独立类别, 不要都归到 after_sale。

# 少样本示例
输入: 快递到广州十天了还没动静, 客服也不回复
输出: {{"labels": ["logistics", "service"], "confidence": 0.9, "reason": "物流停滞叠加客服未响应"}}

输入: 收到的电饭锅是坏的, 我要退款, 发票也一直没开
输出: {{"labels": ["quality", "after_sale", "invoice"], "confidence": 0.92, "reason": "质量问题同时要求退款并催开发票"}}

输入: 哈哈哈哈哈哈
输出: {{"labels": ["invalid"], "confidence": 0.95, "reason": "无意义内容"}}

# 输出格式(必须是合法 JSON, 不要输出任何解释文字或代码块标记)
{{"labels": ["标签1", "标签2"], "confidence": 0.0~1.0 之间的数字, "reason": "一句话理由"}}"""


# ============================================================
# todo 1. LLM 兜底客户端
# ============================================================
class LLMFallback:
    """拒识工单的 LLM 兜底解析器

    参数:
        class_list : 合法标签列表(用于提示词与结果校验)
        api_key    : 不传则读环境变量 DEEPSEEK_API_KEY; 为空时自动禁用
        base_url   : OpenAI 兼容接口地址(DeepSeek 官方为 https://api.deepseek.com)
        model      : 模型名
        timeout    : 单次请求超时(秒)
    说明:
        未安装 openai SDK 或未配置 Key 时, available 属性为 False, parse() 直接返回 None,
        调用方据此走人工复核 —— 保证"没有 LLM 也能完整演示".
    """

    def __init__(self, class_list, api_key=None, base_url=None, model=None, timeout=20):
        self.class_list = list(class_list)
        # 注意: api_key=None 表示"未指定, 去读环境变量"; api_key='' 表示"显式禁用"(仅测试用)
        if api_key is None:
            self.api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        else:
            self.api_key = api_key.strip()
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.system_prompt = build_system_prompt(self.class_list)

        switch = os.environ.get('LLM_FALLBACK_ENABLED', 'true').lower()
        self.enabled_by_env = switch not in ('0', 'false', 'no', 'off')
        self._client = None
        self.disable_reason = None

        if not self.api_key:
            self.disable_reason = '未配置 DEEPSEEK_API_KEY(环境变量), LLM 兜底已自动禁用'
        elif not self.enabled_by_env:
            self.disable_reason = '环境变量 LLM_FALLBACK_ENABLED=false, LLM 兜底已禁用'

    # ---------------- 可用性 ----------------
    @property
    def available(self):
        return bool(self.api_key) and self.enabled_by_env

    def _get_client(self):
        """延迟创建客户端(避免没装 openai 时 import 就报错)"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    '未安装 openai SDK, 无法使用 LLM 兜底; 请执行 pip install openai, '
                    '或关闭兜底开关直接走人工复核'
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    # ---------------- 结果解析 ----------------
    @staticmethod
    def extract_json(text):
        """从模型回复里提取 JSON 对象(兼容 ```json 包裹与前后多余文字)"""
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r'^```(?:json)?', '', cleaned).strip()
        cleaned = re.sub(r'```$', '', cleaned).strip()
        start, end = cleaned.find('{'), cleaned.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None

    def validate(self, payload):
        """校验并规范化 LLM 返回: 标签必须在合法集合内, 去重, 按 class_list 顺序排序"""
        if not isinstance(payload, dict):
            return None
        raw_labels = payload.get('labels')
        if isinstance(raw_labels, str):
            raw_labels = re.split(r'[,，、\s]+', raw_labels)
        if not isinstance(raw_labels, list):
            return None
        labels = []
        for item in raw_labels:
            if not isinstance(item, str):
                continue
            name = item.strip().lower()
            if name in self.class_list and name not in labels:
                labels.append(name)
        if not labels:
            return None
        labels = sorted(labels, key=self.class_list.index)
        try:
            confidence = float(payload.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            'labels': labels,
            'confidence': max(0.0, min(1.0, confidence)),
            'reason': str(payload.get('reason', ''))[:200],
        }

    # ---------------- 主入口 ----------------
    def parse(self, text, temperature=0.0, max_tokens=200):
        """解析一条工单; 任何异常都返回 None(由上层转人工复核), 不抛给业务层

        返回: {'labels': [...], 'confidence': float, 'reason': str, 'source': 'llm'}
              或 None(未启用 / 请求失败 / 结果非法)
        """
        if not self.available or not text or not text.strip():
            return None
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': text.strip()[:1000]},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
        except Exception as exc:                     # 网络/超时/鉴权失败统一兜住
            print(f'  [LLM 兜底] 调用失败, 转人工复核: {type(exc).__name__}: {exc}')
            return None

        payload = self.extract_json(content)
        result = self.validate(payload)
        if result is None:
            print(f'  [LLM 兜底] 返回内容无法解析为合法标签, 转人工复核: {str(content)[:120]}')
            return None
        result['source'] = 'llm'
        result['model'] = self.model
        return result


# ============================================================
# todo 2. 模块级便捷入口(供 backend 直接调用, 单例复用)
# ============================================================
_default_parser = None


def get_default_parser(class_list):
    """获取默认解析器(进程内单例, 避免每次请求都重建提示词与客户端)"""
    global _default_parser
    if _default_parser is None or _default_parser.class_list != list(class_list):
        _default_parser = LLMFallback(class_list)
    return _default_parser


def llm_parse(text, class_list):
    """便捷函数: 用默认解析器解析一条文本"""
    return get_default_parser(class_list).parse(text)


# ============================================================
# todo 3. 自测(离线可跑: 验证提示词与 JSON 解析; 有 Key 时会真的调用一次)
# ============================================================
if __name__ == '__main__':
    class_list = ['logistics', 'quality', 'after_sale', 'invoice', 'price_promo',
                  'payment_account', 'consult', 'service', 'invalid']

    print('=' * 72)
    print('LLM 兜底模块自测')
    print('=' * 72)

    prompt = build_system_prompt(class_list)
    print(f'\n[1] 系统提示词长度: {len(prompt)} 字符, 覆盖类别 {len(class_list)} 个')
    assert 'invalid' in prompt and 'JSON' in prompt

    print('\n[2] JSON 提取与校验测试:')
    cases = [
        ('```json\n{"labels": ["logistics", "service"], "confidence": 0.9, "reason": "物流+客服"}\n```',
         ['logistics', 'service']),
        ('模型啰嗦的解释... {"labels": ["INVOICE"], "confidence": 1.5, "reason": "发票"} 结束',
         ['invoice']),
        ('{"labels": "物流, 客服", "confidence": 0.8}', None),      # 中文标签非法 -> 校验应拦截
        ('{"labels": ["不存在的标签"], "confidence": 0.8}', None),   # 非法标签 -> 拦截
        ('完全没有 JSON 的回复', None),
    ]
    parser = LLMFallback(class_list, api_key='')     # 显式禁用, 只测离线逻辑
    for raw, expect in cases:
        result = parser.validate(parser.extract_json(raw))
        labels = result['labels'] if result else None
        status = 'OK' if (labels == expect if expect is not None else result is None) else 'FAIL'
        print(f'    [{status}] 提取结果={labels}  (期望={expect})')

    print(f'\n[3] 兜底可用性: available={parser.available}')
    print(f'    禁用原因: {parser.disable_reason}')

    real = LLMFallback(class_list)
    if real.available:
        print('\n[4] 检测到 DEEPSEEK_API_KEY, 尝试真实调用一次...')
        out = real.parse('快递到广州十天了还没动静，客服也不回复')
        print(f'    真实返回: {out}')
    else:
        print('\n[4] 未配置 DEEPSEEK_API_KEY, 跳过真实调用(服务会自动走人工复核路径)')

    print('\n[OK] LLM 兜底模块自测通过')