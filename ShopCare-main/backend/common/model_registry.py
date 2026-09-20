"""
ShopCare 模型注册表 (backend/common/model_registry.py)

职责:
    把 02-rf / 03-fasttext / 04-bert 三套模型统一成一个可切换的字典:
        registry.predict(text, model='bert') -> 统一的业务决策字典
    页面上的"模型选择"下拉框点下去, 最终就是改了这里的 model 参数.

一个必须解决的问题: **三套模型的目录里有同名模块**
    02-rf/config.py、03-fasttext/config.py、04-bert/config.py 都叫 config;
    predict_fun 也各自叫 rf_predict_fun / ft_predict_fun / bert_predict_fun(名字不同, 但都从 config 导入).
    在同一个 Python 进程里, 第一个被导入的 config 会一直留在 sys.modules 里,
    后面再 `from config import FTConfig` 拿到的其实是 RF 的 Config —— 不报错, 但参数全错.
    这是"多个独立可运行的脚本目录被同一个服务复用"时的经典陷阱.

    解决办法(_isolated_import): 加载某个阶段的模块时, 临时把它的目录放进 sys.path,
    并把可能冲突的同名模块从 sys.modules 里摘出去, 加载完再原样恢复.
    这样每个阶段仍然保持"能独立运行"的写法, 后端也不需要去改它们的 import.

懒加载与降级:
    * 只有真正用到某个模型时才去 import 它(避免为了用 RF 而被迫加载 torch);
    * 模型文件缺失不会让服务启动失败, 只是该模型在 /models 里显示 unavailable;
    * 请求的模型不可用时, 自动按 MODEL_AVAILABLE 的顺序找一个可用的顶上,
      并在响应里用 fallback_from 说明"你点的模型没用上, 实际用的是谁" —— 不静默替换.
"""

import importlib.util
import os
import sys
import time

from backend.common.config import get_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 阶段目录 -> (配置类所在的文件名, 配置类名, 预测器文件, 预测器类名)
STAGE_SPEC = {
    'rf': ('02-rf', 'config.py', 'RFConfig', 'rf_predict_fun.py', 'RFPredictor'),
    'fasttext': ('03-fasttext', 'config.py', 'FTConfig', 'ft_predict_fun.py', 'FTPredictor'),
    'bert': ('04-bert', 'config.py', 'Config', 'bert_predict_fun.py', 'TicketPredictor'),
}

# 展示用元信息(前端"模型选择"里显示的中文说明)
MODEL_DISPLAY = {
    'rf': {
        'name': 'rf-tfidf',
        'cn': '随机森林 (TF-IDF)',
        'desc': '传统基线: 字面 n-gram 特征 + 9 个二分类森林。训练秒级、完全可解释, 但换种说法容易漏判。',
        'speed': '最快', 'cost': '最低',
    },
    'fasttext': {
        'name': 'fasttext',
        'cn': 'FastText (词向量)',
        'desc': '中间档: 词向量 + n-gram + 线性分类器。有浅层语义, 训练几秒, 单条推理毫秒级。',
        'speed': '快', 'cost': '低',
    },
    'bert': {
        'name': 'bert-lora',
        'cn': 'BERT + LoRA',
        'desc': '效果上限最高: 预训练语义 + LoRA 微调, 对同义改写最鲁棒; 需要模型文件与更多算力。',
        'speed': '较慢', 'cost': '较高',
    },
}


def _isolated_import(stage_dir, filename, module_attr, extra_conflict_names=()):
    """在隔离环境里按文件路径加载模块并取出指定属性

    extra_conflict_names 用于把该阶段里"和其他阶段重名"的模块名一并隔离.
    """
    conflict = set(extra_conflict_names) | {'config', filename[:-3]}
    saved = {}
    for name in list(conflict):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    abs_dir = os.path.join(PROJECT_ROOT, stage_dir)
    uniq_name = '_shopcare_{}_{}'.format(stage_dir.replace('-', '_'), filename[:-3])
    sys.path.insert(0, abs_dir)
    try:
        spec = importlib.util.spec_from_file_location(uniq_name, os.path.join(abs_dir, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[uniq_name] = module      # 必须先注册, 模块内部的导入链条才找得到自己
        spec.loader.exec_module(module)
        return getattr(module, module_attr)
    finally:
        if abs_dir in sys.path:
            sys.path.remove(abs_dir)
        for name in list(conflict):          # 归还借用的模块名
            sys.modules.pop(name, None)
        sys.modules.update(saved)


class ModelRegistry:
    """三套模型的统一入口(懒加载 + 自动降级)"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self._predictors = {}            # model_key -> (predictor 实例, 加载错误)
        self._load_errors = {}

    # ---------------- 可用性 ----------------
    def configured(self):
        """配置里允许使用的模型(按 MODEL_AVAILABLE 的顺序)"""
        return [m for m in self.cfg.model_available if m in STAGE_SPEC]

    def _model_file_ok(self, key):
        """快速检查模型产物是否存在(不加载)"""
        checks = {
            'rf': os.path.join(PROJECT_ROOT, '02-rf', 'save_models', 'rf_tfidf.joblib'),
            'fasttext': os.path.join(PROJECT_ROOT, '03-fasttext', 'save_models', 'fasttext_multilabel.bin'),
            'bert': os.path.join(PROJECT_ROOT, '04-bert', 'save_models', 'bert_lora_multilabel.pt'),
        }
        return os.path.exists(checks.get(key, '')), checks.get(key, '')

    def bert_pretrained_ok(self):
        d = os.path.join(PROJECT_ROOT, '04-bert', 'bert-base-chinese')
        return os.path.exists(os.path.join(d, 'config.json')) and (
            os.path.exists(os.path.join(d, 'pytorch_model.bin'))
            or os.path.exists(os.path.join(d, 'model.safetensors')))

    def models(self):
        """给 /models 接口用的清单(不触发加载, 只看文件与配置)"""
        out = []
        for key in self.configured():
            stage, cfg_file, cfg_cls, pred_file, pred_cls = STAGE_SPEC[key]
            exists, path = self._model_file_ok(key)
            display = dict(MODEL_DISPLAY.get(key, {}))
            display.update({
                'key': key,
                'available': bool(exists),
                'model_path': path,
                'model_exists': exists,
                'is_default': key == self.cfg.default_model,
                'loaded': key in self._predictors,
                'load_error': self._load_errors.get(key),
            })
            if key == 'bert' and not self.bert_pretrained_ok():
                display['available'] = False
                display['extra_hint'] = ('缺少本地预训练模型目录 04-bert/bert-base-chinese '
                                         '(需含 config.json 与权重文件); 本项目不联网下载模型')
            out.append(display)
        return out

    # ---------------- 加载 ----------------
    def _build(self, key):
        stage, cfg_file, cfg_cls, pred_file, pred_cls = STAGE_SPEC[key]
        config_cls = _isolated_import(stage, cfg_file, cfg_cls,
                                      extra_conflict_names=(pred_file[:-3],))
        predictor_cls = _isolated_import(stage, pred_file, pred_cls,
                                         extra_conflict_names=(cfg_file[:-3],))
        cfg = config_cls()
        # 阈值以环境变量/后端配置为准(前端调阈值时改的就是这里)
        try:
            cfg.label_threshold = self.cfg.label_threshold
            cfg.global_threshold = self.cfg.global_threshold
        except (AttributeError, TypeError):
            pass
        return predictor_cls(cfg, enable_llm=self.cfg.enable_llm_fallback)

    def get(self, key, force=False):
        """获取某个模型的预测器(懒加载); 不可用时返回 (None, 原因)"""
        key = (key or self.cfg.default_model).lower()
        if key not in STAGE_SPEC:
            return None, f'未知模型: {key} (可选: {list(STAGE_SPEC)})'
        if key in self._predictors and not force:
            return self._predictors[key], None
        if key in self._load_errors and not force:
            return None, self._load_errors[key]
        try:
            predictor = self._build(key)
            # 立刻尝试加载模型文件, 失败就记下原因(不抛)
            predictor.load()
            self._predictors[key] = predictor
            if getattr(predictor, '_loaded', False):
                self._load_errors.pop(key, None)
            else:
                self._load_errors[key] = getattr(predictor, 'load_error', '未知原因')
            return predictor, self._load_errors.get(key)
        except Exception as exc:              # noqa: BLE001
            msg = f'{type(exc).__name__}: {exc}'
            self._load_errors[key] = msg
            return None, msg

    def resolve(self, requested=None):
        """把"用户点的模型"解析成"实际可用的模型", 返回 (key, predictor, fallback_from)"""
        order = []
        requested = (requested or self.cfg.default_model).lower()
        if requested in STAGE_SPEC:
            order.append(requested)
        for key in self.configured():
            if key not in order:
                order.append(key)
        errors = []
        for idx, key in enumerate(order):
            predictor, error = self.get(key)
            if predictor is not None and getattr(predictor, '_loaded', False):
                return key, predictor, (requested if idx > 0 and key != requested else None)
            errors.append(f'{key}: {error}')
        raise RuntimeError('所有模型都不可用: ' + ' | '.join(errors))

    def status(self):
        """给 /health 用的状态汇总"""
        out = {}
        for key in STAGE_SPEC:
            predictor = self._predictors.get(key)
            out[key] = predictor.status if predictor is not None else {
                'model': MODEL_DISPLAY.get(key, {}).get('name', key),
                'loaded': False,
                'load_error': self._load_errors.get(key),
            }
        out['default_model'] = self.cfg.default_model
        out['available'] = [m['key'] for m in self.models() if m['available']]
        return out

    # ---------------- 推理 ----------------
    def predict(self, text, model=None, use_llm_fallback=None, top_k=3):
        """统一推理入口

        返回: (决策字典, 实际使用的模型 key, 是否发生了降级)
        """
        if use_llm_fallback is None:
            use_llm_fallback = self.cfg.enable_llm_fallback
        key, predictor, fallback_from = self.resolve(model)
        result = predictor.predict(text, use_llm_fallback=use_llm_fallback, top_k=top_k)
        result['model_used'] = key
        if fallback_from:
            result['fallback_from'] = fallback_from
        return result, key, fallback_from


_registry = None


def get_registry(cfg=None):
    global _registry
    if _registry is None:
        _registry = ModelRegistry(cfg)
    return _registry


if __name__ == '__main__':
    print('=' * 72)
    print('ShopCare model_registry 自测')
    print('=' * 72)
    reg = get_registry()
    print('\n配置允许的模型:', reg.configured())
    print('\n模型清单(不触发加载):')
    for m in reg.models():
        print('  {:<10}{:<20}available={:<6}{}'.format(
            m['key'], m.get('cn', ''), str(m['available']), m.get('extra_hint', '')))

    # 关键验证: 三套模型的 config 类必须**互相独立**, 不能被同名模块串味
    print('\n隔离加载验证(三个阶段的 config 类各自独立):')
    seen = {}
    for key, (stage, cfg_file, cfg_cls, _, _) in STAGE_SPEC.items():
        cls = _isolated_import(stage, cfg_file, cfg_cls)
        seen[key] = cls
        print('  {:<10}{:<16}module={}'.format(key, cls.__name__, cls.__module__))
    assert len({id(v) for v in seen.values()}) == 3, '三个 config 类不能是同一个对象'
    assert seen['rf'].__name__ == 'RFConfig'
    assert seen['fasttext'].__name__ == 'FTConfig'
    assert seen['bert'].__name__ == 'Config'
    # 实例化后各自的关键属性要正确, 这是"串味"最容易暴露的地方
    rf, ft = seen['rf'](), seen['fasttext']()
    assert hasattr(rf, 'n_estimators') and not hasattr(rf, 'dim'), 'RF 配置不该有 FastText 的 dim'
    assert hasattr(ft, 'dim') and not hasattr(ft, 'n_estimators'), 'FastText 配置不该有 RF 的 n_estimators'
    assert rf.model_name == 'rf-tfidf' and ft.model_name == 'fasttext'
    print('  [OK] 三份配置互不串味 (RF.model_name=%s, FT.model_name=%s)' % (rf.model_name, ft.model_name))

    # 加载 + 推理(模型文件缺失时应优雅报告, 不崩)
    print('\n尝试加载并推理:')
    ok_any = False
    for key in reg.configured():
        t0 = time.time()
        predictor, err = reg.get(key)
        if predictor is not None and getattr(predictor, '_loaded', False):
            r = predictor.predict('快递一直没到, 客服也不回复')
            print('  {:<10}加载 {:.2f}s -> {} | labels={} | 拒识={}'.format(
                key, time.time() - t0,
                r.get('model_used'), [x['label'] for x in r['labels']], r['rejected']))
            ok_any = True
        else:
            print('  {:<10}不可用: {}'.format(key, err))
    if not ok_any:
        print('  [提示] 没有任何模型可用, 请先训练: 02-rf/rf_train.py 或 03-fasttext/ft_train.py')
    print('\n[OK] model_registry 自测通过')