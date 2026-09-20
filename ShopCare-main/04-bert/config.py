"""
ShopCare 核心模型配置 (04-bert)

模块说明:
    集中管理 04-bert 阶段的所有路径、模型超参数、LoRA 参数与拒识阈值,
    供训练脚本(train_bert.py)、推理入口(bert_predict_fun.py)、
    以及复用它的 05/06/07 优化阶段统一调用.

设计说明(与参考项目 MindCare 的一个关键差异):
    本 Config **不在 __init__ 里加载 BERT 权重**, 只保存路径与超参数.
    原因: 读一个路径不该触发几百 MB 的模型加载; 而且模型目录还没放好时,
    import 本文件不应该直接崩. 真正加载模型统一走 load_pretrained() ——
    该函数**强制本地目录加载, 绝不联网下载**.

运行方式:
    python 04-bert/config.py     # 打印当前配置摘要
"""

import os

# 项目根目录: 本文件位于 04-bert/ 下, 向上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """04-bert 全局配置: 路径 / 数据 / 模型 / LoRA / 训练 / 拒识阈值"""

    def __init__(self):
        # ==================== todo 1. 基础路径 ====================
        self.root_path = PROJECT_ROOT.replace('\\', '/') + '/'
        self.model_name = 'bert-lora'

        # 数据路径(与 01-data/data_format.md 一致)
        self.train_path = os.path.join(PROJECT_ROOT, '01-data', 'train.txt')
        self.dev_path = os.path.join(PROJECT_ROOT, '01-data', 'dev.txt')
        self.test_path = os.path.join(PROJECT_ROOT, '01-data', 'test.txt')
        self.class_path = os.path.join(PROJECT_ROOT, '01-data', 'class.txt')

        # 模型与产物路径
        self.save_dir = os.path.join(PROJECT_ROOT, '04-bert', 'save_models')
        self.result_dir = os.path.join(PROJECT_ROOT, '04-bert', 'result')
        self.model_save_path = os.path.join(self.save_dir, 'bert_lora_multilabel.pt')
        self.pos_weight_path = os.path.join(self.save_dir, 'pos_weight.pt')
        self.train_log_path = os.path.join(self.result_dir, 'train_log.json')
        self.hard_example_log_path = os.path.join(self.result_dir, 'hard_example_log.json')

        # ==================== todo 2. 类别体系 ====================
        # 每行一个标签, 行号即索引(第 1 行索引 0)
        self.class_list = [line.strip() for line in open(self.class_path, encoding='utf-8') if line.strip()]
        self.num_labels = len(self.class_list)
        self.id2class = {i: c for i, c in enumerate(self.class_list)}
        self.class2id = {c: i for i, c in enumerate(self.class_list)}

        # ==================== todo 3. 预训练模型(本地, 不联网) ====================
        # 可用环境变量 BERT_MODEL_DIR 指向任意本地模型目录(例如换成 RoBERTa)
        self.bert_dir = os.environ.get(
            'BERT_MODEL_DIR',
            os.path.join(PROJECT_ROOT, '04-bert', 'bert-base-chinese')
        )
        self.max_len = 96          # 工单多为短文本, 96 足够; 由 data_eda 的截断比例决定是否需要调大

        # ==================== todo 4. LoRA 参数 ====================
        self.use_lora = True                 # 关掉即退化为"冻结主干 + 只训分类头"的对照组
        self.lora_r = 8                      # 低秩矩阵的秩 r
        self.lora_alpha = 16                 # 缩放系数 alpha, 实际缩放 = alpha / r
        self.lora_dropout = 0.1
        self.lora_target_modules = ['query', 'value']   # 注入注意力的 Q/V 投影

        # ==================== todo 5. 训练超参数 ====================
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 2e-4            # LoRA 专用学习率(比全量微调高一个量级)
        self.weight_decay = 0.01
        self.dropout = 0.3                   # 分类头前的 dropout
        self.patience = 2                    # 早停: dev Micro-F1 连续 N 轮不升则停
        self.seed = 42
        self.num_workers = 0                 # Windows 下建议 0, 避免多进程 DataLoader 问题

        # ==================== todo 6. 多标签难例动态采样(创新点 1) ====================
        self.use_hard_sampling = True
        self.hard_w_label_cnt = 0.5          # 标签数信号权重
        self.hard_w_miss = 1.0               # 漏判率信号权重
        self.hard_w_conf = 0.8               # 置信度缺口信号权重
        self.hard_w_max = 5.0                # 单样本权重上限(防止个别难例主导训练)

        # ==================== todo 7. 双阈值拒识(创新点 2) ====================
        self.label_threshold = float(os.environ.get('LABEL_THRESHOLD', 0.5))    # 单标签激活阈值
        self.global_threshold = float(os.environ.get('GLOBAL_THRESHOLD', 0.8))  # 全局平均置信度阈值

        # ==================== todo 8. 设备 ====================
        # 延迟导入 torch: 让"只想看配置"的场景不必装 torch
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'

    # ------------------------------------------------------------------
    def check_files(self, need_model=True):
        """启动前的自检: 数据与模型目录是否存在, 缺失时给出明确的修复提示"""
        problems = []
        for name, path in [('训练集', self.train_path), ('验证集', self.dev_path),
                           ('类别文件', self.class_path)]:
            if not os.path.exists(path):
                problems.append(f'{name}缺失: {path}')
        if need_model and not os.path.isdir(self.bert_dir):
            problems.append(
                f'预训练模型目录缺失: {self.bert_dir}\n'
                f'    本项目不自动联网下载模型, 请手动放置 bert-base-chinese 目录, 或设置环境变量 BERT_MODEL_DIR'
            )
        elif need_model and not os.path.exists(os.path.join(self.bert_dir, 'config.json')):
            problems.append(f'模型目录里缺少 config.json: {self.bert_dir}')
        return problems

    def summary(self):
        """返回配置摘要文本(训练开始时打印, 便于复现实验)"""
        return '\n'.join([
            f'  项目根目录   : {self.root_path}',
            f'  标签体系     : {self.num_labels} 类 -> {", ".join(self.class_list)}',
            f'  预训练模型   : {self.bert_dir}',
            f'  最大长度     : {self.max_len}',
            f'  LoRA         : r={self.lora_r}, alpha={self.lora_alpha}, '
            f'dropout={self.lora_dropout}, target={self.lora_target_modules}',
            f'  训练         : epochs={self.epochs}, batch={self.batch_size}, '
            f'lr={self.learning_rate}, patience={self.patience}',
            f'  难例动态采样 : {self.use_hard_sampling} '
            f'(α={self.hard_w_label_cnt}, β={self.hard_w_miss}, γ={self.hard_w_conf})',
            f'  拒识阈值     : 单标签 {self.label_threshold} / 全局 {self.global_threshold}',
            f'  设备         : {self.device}',
        ])


if __name__ == '__main__':
    c = Config()
    print('=' * 72)
    print('ShopCare 04-bert 配置摘要')
    print('=' * 72)
    print(c.summary())
    problems = c.check_files(need_model=False)
    if problems:
        print('\n[警告] 文件自检发现问题:')
        for p in problems:
            print('  - ' + p)
    else:
        print('\n[OK] 文件自检通过')
