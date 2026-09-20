"""
ShopCare 中文分词/清洗工具 (tools)

用于 02-rf(TF-IDF 特征) 与 03-fasttext(词级输入) 的文本预处理。

为什么单独放一个文件而不是写在各自的 train 脚本里:
    sklearn 的 TfidfVectorizer 会把 tokenizer **函数对象**一起序列化进模型文件.
    pickle 对"定义在 __main__ 里的函数"是按值序列化的, 换个脚本加载就找不到函数了;
    定义在一个稳定的模块里则是**按引用**序列化(--> module.func), 加载时才可靠.
    所以 tokenizer 必须是模块级函数, 这也顺便让两套模型共用同一份分词口径.

口径说明:
    * 工单文本口语化、含订单号/手机号/金额, 这些数字对分类是噪声甚至有害
      (模型会记住"某个订单号 -> 某个标签"), 所以统一做**占位符替换**而不是直接删
      —— 保留"这里有个数字"这个信息, 但不泄露具体值.
    * 中文不做词干化, 也不用去停用词: TF-IDF 的 idf 权重会自动压制高频无信息词,
      手动去停用词反而容易误删"不""没""退"这类决定标签的关键否定词。
"""

import re

# ------------------------------------------------------------
# 正则: 先做结构化占位, 再做分词
# ------------------------------------------------------------
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_RE_PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
_RE_ORDER = re.compile(r'(?<![A-Za-z0-9])\d{8,}(?![A-Za-z0-9])')      # 订单号/快递单号
_RE_MONEY = re.compile(r'\d+(?:\.\d+)?\s*(?:元|块|圆|¥|RMB)')
_RE_NUM = re.compile(r'\d+(?:\.\d+)?')

_RE_SPACE = re.compile(r'\s+')
# 只保留中文/字母/数字和 [占位符] 的方括号, 其余(emoji、乱码)统一丢弃
_RE_KEEP = re.compile(r'[^\u4e00-\u9fffA-Za-z0-9\[\]]')
# 三连以上的重复字符压缩为两个: "啊啊啊啊啊" -> "啊啊" (保留强调语气但降低噪声)
_RE_REPEAT = re.compile(r'(.)\1{2,}')


def clean_text(text, keep_symbols=False):
    """把一条工单文本规范化成干净的字符串

    参数 keep_symbols: True 时保留标点(给 BERT 用更自然); False 时只留中日韩汉字/字母/数字
    """
    if not text:
        return ''
    s = str(text)
    s = _RE_URL.sub(' [链接] ', s)
    s = _RE_EMAIL.sub(' [邮箱] ', s)
    s = _RE_PHONE.sub(' [手机号] ', s)
    s = _RE_MONEY.sub(' [金额] ', s)
    s = _RE_ORDER.sub(' [单号] ', s)
    s = _RE_NUM.sub(' [数字] ', s)
    s = _RE_REPEAT.sub(r'\1\1', s)
    s = _RE_SPACE.sub(' ', s).strip()
    if not keep_symbols:
        s = _RE_KEEP.sub(' ', s)
        s = _RE_SPACE.sub(' ', s).strip()
    return s


def has_jieba():
    """检测本环境是否装了 jieba(没装就自动降级到字符级)"""
    try:
        import jieba  # noqa: F401
        return True
    except ImportError:
        return False


def jieba_tokenize(text):
    """词级切分(需要 jieba). 作为 TfidfVectorizer 的 tokenizer 时必须返回 list[str]"""
    import jieba
    return [w for w in jieba.lcut(text) if w.strip()]


def char_tokenize(text):
    """字符级切分: 中文按单字(由 ngram_range=(2,4) 组合成词组), 英文/数字按整体保留

    这么切有个好处: 不需要任何词典, 对"快递""快递员""快递费"这类未登录词和错别字
    ('快弟') 都天然鲁棒 —— 这也是 02-rf 作为强基线的意义所在。
    """
    if not text:
        return []
    tokens, buf = [], ''
    for ch in text:
        if re.match(r'[A-Za-z0-9]', ch):
            buf += ch
            continue
        if buf:
            tokens.append(buf)
            buf = ''
        # 只把汉字当作单字 token; 方括号/标点等一律丢弃(占位符的内容本身已是汉字)
        if '\u4e00' <= ch <= '\u9fff':
            tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


def get_tokenizer(mode='auto'):
    """按模式返回 (实际模式, tokenizer 函数)

    mode: 'word' 强制词级 / 'char' 强制字符级 / 'auto' 有 jieba 用词级, 否则字符级
    """
    if mode == 'word':
        if not has_jieba():
            raise ImportError('feature_mode=word 需要 jieba, 请先安装: pip install jieba')
        return 'word', jieba_tokenize
    if mode == 'char':
        return 'char', char_tokenize
    if mode == 'auto':
        return ('word', jieba_tokenize) if has_jieba() else ('char', char_tokenize)
    raise ValueError(f'未知的 feature_mode: {mode} (可选 word / char / auto)')


if __name__ == '__main__':
    print('=' * 72)
    print('text_tokenize 自检')
    print('=' * 72)
    raw = '我的订单 2024091500123 花了 199.5元 买的羽绒服，快递到广州十天了还没动静！！！客服电话13812345678'
    cleaned = clean_text(raw)
    print('原文   :', raw)
    print('清洗后 :', cleaned)
    assert '[单号]' in cleaned and '[金额]' in cleaned and '[手机号]' in cleaned, cleaned
    print('\n字符级 :', char_tokenize(cleaned)[:20], '...共', len(char_tokenize(cleaned)), '个 token')

    mode, tok = get_tokenizer('auto')
    print(f'\n分辨率 : mode={mode}, jieba 可用={has_jieba()}')
    print('分词结果:', tok(cleaned)[:20])

    # 重复字符压缩与乱码清理
    print('\n重复压缩:', clean_text('啊啊啊啊啊好烦啊啊啊'))
    print('乱码清理:', clean_text('***###快递呢？？？'))
    print('\n[OK] text_tokenize 自检通过')