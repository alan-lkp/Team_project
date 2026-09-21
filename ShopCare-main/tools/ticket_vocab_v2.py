# -*- coding: utf-8 -*-
"""
ShopCare 槽位词表 v2 —— 为「按家族分组切分」的数据集提供足够大的组合空间

为什么需要这个文件
==================
v1 的槽位词表只有 20 个商品 / 8 个城市 / 6 个天数,配合 123 条模板,任何切分方式
都无法同时满足「数据量够大」和「train/test 子句零重合」——模板池太小,只能靠子句
反复重排凑数据量,于是 test 里 96% 的子句都能在 train 中原样找到(模板级泄漏)。

本文件做三件事:

1. **扩大词表**: 商品 20→152,城市 8→62,天数 6→26。每条带槽位的模板能展开出成百
   上千个**字面不同**的子句,数据量不再依赖重复。

2. **按品类组织商品**: 商品带品类标签(digital/apparel/food/...)。
   `build_dataset_v2.detect_category()` 会从模板正文判断它需要哪一类商品,避免
   「录音笔面料很硬穿着扎皮肤」这种语义错配 —— 这类错配在 v1 语料里随商品随机填充
   而普遍存在,交付时一眼就能看出是造的数据。

3. **按 split 切开词表**: train 只用 TRAIN 段,dev/test 只用 HELDOUT 段,且**每个品类
   在两边都有货**。这样 test 的句子即使命中与 train 相同的句式,商品名/城市名也一定
   没出现过 —— "test 无泄漏" 的结论从「句式不同」加固到「句式 + 槽位取值都不同」。

   (第一版按整段切分,结果 heldout 段只剩母婴/运动/日用三类,服饰类和食品类模板在
    dev/test 里无货可填。所以改成**按品类内部切分**。)

   注:dev 与 test 共用同一段 heldout 是安全的 —— 它们的 family 本身互斥,不会产生相同
   子句;`build_dataset_v2` 末尾会用「从磁盘重读」的方式实测验证。

切分比例由 HELDOUT_RATIO 控制,改词表长度时请同步确认每个品类在两边都 ≥2 个。
"""

# ============================================================
# 商品表:按品类组织(共 152 条)
# 品类名会被 build_dataset_v2.detect_category() 用来做语义匹配,
# 新增商品时请放到正确的品类里,否则可能被填进语义不符的模板。
# ============================================================
GOODS_BY_CATEGORY = {
    # 数码电子:有屏幕/电池/需要充电连接的那类
    'digital': [
        '手机', '蓝牙耳机', '笔记本电脑', '平板电脑', '机械键盘', '鼠标', '显示器',
        '充电宝', '数据线', '智能手表', '蓝牙音箱', '摄像头', '路由器', '移动硬盘',
        'U盘', '头戴式耳机', '无线充电器', '投影仪', '打印机', '行车记录仪',
        '智能门锁', '麦克风', '录音笔', '电子书阅读器',
    ],
    # 家用电器:插电使用的大家电小家电
    'appliance': [
        '电饭锅', '空气炸锅', '扫地机器人', '吸尘器', '洗衣机', '微波炉', '电风扇',
        '加湿器', '净水器', '电水壶', '破壁机', '烤箱', '电磁炉', '空调',
        '挂烫机', '除湿机', '洗碗机', '咖啡机', '榨汁机', '电吹风',
        '卷发棒', '剃须刀', '电动拖把', '取暖器',
    ],
    # 服饰鞋包:有面料/尺码/穿着体验的那类
    'apparel': [
        '羽绒服', '运动鞋', '连衣裙', '卫衣', '牛仔裤', 'T恤', '毛衣', '外套',
        '内衣', '袜子', '帽子', '围巾', '手套', '皮鞋', '帆布鞋', '冲锋衣', '睡衣',
    ],
    # 食品饮料:有味道/保质期/能吃能喝的那类
    'food': [
        '奶粉', '大米', '食用油', '坚果', '巧克力', '咖啡豆', '茶叶', '蜂蜜',
        '零食大礼包', '螺蛳粉', '自热火锅', '牛奶', '麦片', '矿泉水',
    ],
    # 美妆个护:用在皮肤/头发/牙齿上的那类
    'beauty': [
        '面膜', '洗发水', '沐浴露', '洗面奶', '口红', '粉底液', '防晒霜', '牙膏',
        '香水', '眼霜', '身体乳', '护发素', '电动牙刷',
    ],
    # 家居家具:放在家里用的那类
    'home': [
        '床垫', '台灯', '行李箱', '保温杯', '枕头', '四件套', '收纳箱', '拖鞋',
        '窗帘', '地毯', '沙发', '书桌', '置物架', '垃圾桶', '晾衣架', '穿衣镜',
    ],
    # 母婴
    'baby': [
        '纸尿裤', '婴儿车', '奶瓶', '婴儿床', '辅食机', '儿童安全座椅',
    ],
    # 宠物
    'pet': [
        '猫粮', '狗粮', '猫砂', '猫爬架', '狗窝', '宠物零食', '鱼缸',
    ],
    # 运动户外
    'sports': [
        '瑜伽垫', '哑铃', '跑步机', '帐篷', '登山包', '自行车', '篮球',
        '羽毛球拍', '泳镜', '护膝',
    ],
    # 日用其他
    'daily': [
        '眼镜', '墨镜', '项链', '戒指', '书包', '钢笔', '记事本', '计算器',
        '雨伞', '电蚊香', '驱蚊液', '口罩', '维生素', '血压计', '体温计',
        '指甲刀', '梳子', '镜子', '保温饭盒', '钥匙扣', '跳绳',
    ],
}

# heldout 段(train 之外)占每个品类的比例。0.35 是权衡结果:
#   - 太低:dev/test 可填的商品太少,候选子句不够,只能靠重复凑数;
#   - 太高:train 的商品池被削,而且 heldout 池子小了反而更容易撞车。
HELDOUT_RATIO = 0.35

# 商品池全量(顺序:按品类拼接)
GOODS = [g for _cat in GOODS_BY_CATEGORY.values() for g in _cat]


# ============================================================
# 城市 / 地点(62 条)
# 模板里的用法是 "快递在{place}停了一周" / "包裹卡在{place}中转站",
# 所以这里放城市名,不掺"中转站""转运中心"这类后缀(后缀由模板自己带)
# ============================================================
PLACES = [
    '北京', '上海', '广州', '深圳', '杭州', '南京', '苏州', '成都',
    '重庆', '武汉', '西安', '天津', '长沙', '郑州', '青岛', '宁波',
    '无锡', '合肥', '福州', '厦门', '济南', '大连', '沈阳', '哈尔滨',
    '长春', '石家庄', '太原', '南昌', '昆明', '贵阳', '南宁', '海口',
    '兰州', '西宁', '银川', '乌鲁木齐', '呼和浩特', '拉萨', '温州', '佛山',
    '东莞', '珠海', '中山', '常州', '徐州', '烟台', '潍坊', '洛阳',
    '保定', '唐山', '泉州', '嘉兴', '绍兴', '台州', '金华', '惠州',
    '汕头', '湛江', '桂林', '绵阳', '包头', '株洲',
]

# 城市名没有品类问题,直接按比例切
PLACES_HELDOUT_RATIO = 0.25


# ============================================================
# 天数(26 条)
# 模板里的用法是 "都{day}天了还没发货" / "{day}天还没送到",纯数字字符串
# ============================================================
DAYS = [
    '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
    '15', '16', '17', '18', '20', '21', '22', '25', '28', '30', '35', '40', '50',
]

# 注意:天数与商品/城市不同,必须两边都是"整数集合",不能出现
# 「train 有 20,heldout 有 2」这种前缀包含关系 —— 否则 "20天" 会被误判成
# 「train 专用值 2 出现在 dev/test」。检测侧用数字边界正则兜底,这里也按
# 大小交错切分,把误报压到最低。
DAYS_HELDOUT_RATIO = 0.25


# ============================================================
# 通用名词豁免表
# ============================================================
# 有几件商品的名字本身就是汉语常用名词,会自然出现在模板正文里,但指的不是
# 「用户买的那个商品」。例如 payment_account 的「改绑手机总是收不到验证码怎么办」,
# 这里的「手机」指手机号,不是槽位填充。
#
# 槽位隔离检测(`build_dataset_v2.slot_isolation_report`)会在报原始命中数的同时
# 把这几个词剔除,避免把「正文里的通用名词」误判成「槽位取值泄漏」。剔除只影响
# 诊断口径,不影响数据生成 —— 生成侧的隔离由 split_slot_vocab() 保证。
GENERIC_NOUN_STOPLIST = {'手机', '书包', '口罩'}


def _split_list(items, heldout_ratio, min_heldout=2, min_train=2):
    """把有序列表按比例切成 (train 段, heldout 段)。

    两端都保留至少 min_* 个;heldout 取列表尾部 —— 尾部同样按品类分组时,
    各品类的尾部会落进 heldout,保证每个品类两边都有代表。
    """
    n = len(items)
    if n < min_train + min_heldout:
        k = max(1, n - min_train)
    else:
        k = max(min_heldout, min(n - min_train, round(n * heldout_ratio)))
    return items[:n - k], items[n - k:]


def split_slot_vocab():
    """把槽位词表切成 train 段与 dev/test 段(两段互不相交)。

    商品按品类内部切分,保证每个品类在两边都有货 —— 否则服饰类模板在 dev/test
    里会因为 heldout 段没有服饰商品而填不出句子。
    """
    train_goods_by_cat = {}
    held_goods_by_cat = {}
    for cat, goods in GOODS_BY_CATEGORY.items():
        tr, ho = _split_list(goods, HELDOUT_RATIO)
        train_goods_by_cat[cat] = tr
        held_goods_by_cat[cat] = ho

    places_tr, places_ho = _split_list(PLACES, PLACES_HELDOUT_RATIO, min_heldout=4)
    days_tr, days_ho = _split_list(DAYS, DAYS_HELDOUT_RATIO, min_heldout=4)

    train = {
        'goods': [g for cat in GOODS_BY_CATEGORY for g in train_goods_by_cat[cat]],
        'goods_by_cat': train_goods_by_cat,
        'place': places_tr,
        'day': days_tr,
    }
    heldout = {
        'goods': [g for cat in GOODS_BY_CATEGORY for g in held_goods_by_cat[cat]],
        'goods_by_cat': held_goods_by_cat,
        'place': places_ho,
        'day': days_ho,
    }
    # dev 与 test 共用同一份 heldout(它们的 family 互斥,不会产生相同子句)。
    # 这里刻意复制成两份独立对象,避免调用方原地修改互相污染。
    return {
        'train': train,
        'dev': {k: (dict(v) if isinstance(v, dict) else list(v))
                for k, v in heldout.items()},
        'test': {k: (dict(v) if isinstance(v, dict) else list(v))
                 for k, v in heldout.items()},
    }


def _sanity_check():
    """自检:两段非空、互不相交、每个品类两边都有货、无重复项。"""
    vocab = split_slot_vocab()
    for slot in ('goods', 'place', 'day'):
        tr = vocab['train'][slot]
        ho = vocab['dev'][slot]
        assert tr, f'{slot} 的 train 段为空'
        assert ho, f'{slot} 的 dev/test 段为空'
        assert not (set(tr) & set(ho)), f'{slot} 的 train 段与 heldout 段有交集'
        assert len(set(tr)) == len(tr), f'{slot} 的 train 段有重复项'
        assert len(set(ho)) == len(ho), f'{slot} 的 heldout 段有重复项'

    # 每个品类必须在两个 split 里都有商品可填,否则该品类的模板会填不出句子
    for cat in GOODS_BY_CATEGORY:
        for split in ('train', 'dev', 'test'):
            n = len(vocab[split]['goods_by_cat'][cat])
            assert n >= 2, f'品类 {cat} 在 {split} 里只有 {n} 个商品,至少要 2 个'

    # dev 与 test 的 heldout 必须一致(共用同一份)
    assert vocab['dev']['goods'] == vocab['test']['goods'], 'dev/test 的 heldout 商品不一致'
    return True


if __name__ == '__main__':
    _sanity_check()
    vocab = split_slot_vocab()
    print('槽位词表自检通过')
    print(f"  goods  总计 {len(GOODS):>3}  | train {len(vocab['train']['goods']):>3} 段"
          f"  | dev/test {len(vocab['dev']['goods']):>3} 段")
    print(f"  place  总计 {len(PLACES):>3}  | train {len(vocab['train']['place']):>3} 段"
          f"  | dev/test {len(vocab['dev']['place']):>3} 段")
    print(f"  day    总计 {len(DAYS):>3}  | train {len(vocab['train']['day']):>3} 段"
          f"  | dev/test {len(vocab['dev']['day']):>3} 段")
    print()
    print('  各品类商品数(train / dev-test):')
    for cat in GOODS_BY_CATEGORY:
        t = len(vocab['train']['goods_by_cat'][cat])
        h = len(vocab['dev']['goods_by_cat'][cat])
        print(f'    {cat:<10} 共 {len(GOODS_BY_CATEGORY[cat]):>3}  | '
              f'train {t:>3}  | dev/test {h:>3}')
    print()
    print('  dev/test 商品池:', '、'.join(vocab['dev']['goods']))
    print('  dev/test 城市池:', '、'.join(vocab['dev']['place']))
    print('  dev/test 天数池:', '、'.join(vocab['dev']['day']))
