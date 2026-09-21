"""
BERT + LoRA 多标签训练主脚本 (04-bert)

功能说明:
    一条命令跑完 "读数据 -> 注入LoRA -> 多标签训练(含难例动态加权采样) -> 早停 -> 保存最优 -> 测试集评估"

核心流程:
    1. 读 01-data/train|dev|test.txt, 转 multi-hot 标签;
    2. 统计 pos_weight(负/正样本比), 交给 BCEWithLogitsLoss 治长尾不均衡;
    3. 注入 LoRA(query/value) 并冻结主干, 只训练 LoRA + 分类头(可训练参数 < 1%);
    4. 训练: 每轮结束在 dev 上算 Micro-F1, 连续 patience 轮不升则早停, 只保存最优权重;
    5. 创新点 1: 从第 2 轮开始, 用上一轮模型对训练集重新打分 -> 更新难例权重 -> 重建采样器;
    6. 训练结束用最优权重在 test 集上汇报最终指标(学术 + 业务), 并写入 result/ 目录.

运行方式:
    python 04-bert/train_bert.py                                  # 默认配置
    python 04-bert/train_bert.py --epochs 3 --batch_size 16       # 小显存
    python 04-bert/train_bert.py --use_hard_sampling false        # 消融: 关闭难例采样
    python 04-bert/train_bert.py --dry_run true                   # 只跑 1 个 batch 验证链路

产物:
    04-bert/save_models/bert_lora_multilabel.pt   最优模型(LoRA + 分类头, 含元信息)
    04-bert/save_models/pos_weight.pt             类别权重(复现实验用)
    04-bert/result/train_log.json                 每轮指标
    04-bert/result/hard_example_log.json          难例权重演化轨迹(创新点 1 的证据)
    04-bert/result/test_metrics.json              测试集最终指标
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# 让脚本无论从哪个目录启动, 都能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 项目根目录: 为了用 tools/ 下共享的概率校准与阈值标定(与 02-rf / 03-fasttext 同一套,
# 否则三套模型的"置信度"和"拒识阈值"口径不一致, 对照表就没法比)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from dataloader_utils import build_all_dataloader, compute_pos_weight, load_tokenizer
from hard_example_sampler import HardExampleSampler
from model2dev_utils import (_to_np, compute_metrics, run_inference,
                             save_metrics, print_metrics)
from multilabel_model import BertMultiLabelClassifier, build_loss, build_model
from tools.ml_metrics import grid_search_thresholds                        # noqa: E402
from tools.prob_calibration import (apply_calibrators, fit_calibrators,   # noqa: E402
                                    summary as calibration_summary)


def str2bool(value):
    """命令行布尔参数解析: --use_hard_sampling true/false/1/0"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def set_seed(seed):
    """固定随机种子, 保证实验可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, cfg, epoch, total_epochs):
    """训练一个 epoch, 返回平均损失与耗时"""
    model.train()
    total_loss, steps = 0.0, 0
    start = time.time()
    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(cfg.device)
        attention_mask = batch['attention_mask'].to(cfg.device)
        token_type_ids = batch.get('token_type_ids')
        token_type_ids = token_type_ids.to(cfg.device) if token_type_ids is not None else None
        labels = batch['labels'].to(cfg.device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1
        if step % 50 == 0:
            print(f'    epoch {epoch}/{total_epochs} | step {step}/{len(dataloader)} | '
                  f'loss {total_loss / steps:.4f}')

    return total_loss / max(steps, 1), time.time() - start


def main():
    parser = argparse.ArgumentParser(description='ShopCare BERT+LoRA 多标签训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数(默认取 config)')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小(默认取 config)')
    parser.add_argument('--lr', type=float, default=None, help='学习率(默认取 config)')
    parser.add_argument('--use_lora', type=str, default='true', help='是否使用 LoRA(对照组可设 false)')
    parser.add_argument('--use_hard_sampling', type=str, default='true', help='难例动态加权采样开关')
    parser.add_argument('--dry_run', type=str, default='false', help='只跑 1 个 batch 验证链路')
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        # 注意属性名是 learning_rate —— config 里没有 lr 这个字段。
        # 原来写的是 cfg.lr, 于是 --lr 传了也静默无效(优化器读的是 cfg.learning_rate)。
        cfg.learning_rate = args.lr
    cfg.use_lora = str2bool(args.use_lora)
    cfg.use_hard_sampling = str2bool(args.use_hard_sampling)
    dry_run = str2bool(args.dry_run)

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    print('=' * 72)
    print('ShopCare 04-bert 训练  (BERT + LoRA 多标签 + 难例动态加权采样)')
    print('=' * 72)
    print(cfg.summary())

    # todo 1. 自检: 数据与本地模型目录
    problems = cfg.check_files(need_model=True)
    if problems:
        print('\n[错误] 启动自检未通过:')
        for p in problems:
            print('  - ' + p)
        sys.exit(1)

    # todo 2. 数据与分词器
    tokenizer = load_tokenizer(cfg)
    print('\n[1/6] 加载数据...')
    train_dataset, train_loader, dev_loader, test_loader = build_all_dataloader(cfg, tokenizer)

    # todo 3. 类别权重(治长尾不均衡)
    print('\n[2/6] 统计类别权重 pos_weight...')
    pos_weight = compute_pos_weight(train_dataset)
    torch.save(pos_weight, cfg.pos_weight_path)
    print('  按标签顺序: ' + ', '.join(f'{name}={w:.2f}'
                                     for name, w in zip(cfg.class_list, pos_weight.tolist())))

    # todo 4. 模型 / 损失 / 优化器
    print('\n[3/6] 构建模型...')
    model, criterion = build_model(cfg, pos_weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    try:
        from transformers import get_linear_schedule_with_warmup
        total_steps = max(len(train_loader) * cfg.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        print(f'  学习率调度: 线性 warmup(10%) + 线性衰减, 总步数 {total_steps}')
    except ImportError:
        print('  [提示] 未找到 get_linear_schedule_with_warmup, 使用固定学习率')

    # todo 5. dry run: 只跑一个 batch, 快速验证链路是否通
    if dry_run:
        print('\n[dry_run] 只跑 1 个 batch, 验证前向/反向是否正常...')
        model.train()
        batch = next(iter(train_loader))
        inputs = {k: v.to(cfg.device) for k, v in batch.items()
                  if k in ('input_ids', 'attention_mask', 'token_type_ids')}
        logits = model(**inputs)
        loss = criterion(logits, batch['labels'].to(cfg.device))
        loss.backward()
        print(f'  logits 形状 {tuple(logits.shape)} | loss {loss.item():.4f} | 反向传播正常')
        print('\n[OK] dry_run 通过: 链路可用, 去掉 --dry_run 即可正式训练')
        return

    # todo 6. 训练主循环(含早停 + 难例动态采样)
    print('\n[4/6] 开始训练...')
    hard_sampler = HardExampleSampler(cfg) if cfg.use_hard_sampling else None
    best_micro_f1, best_epoch, no_improve = -1.0, 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # 创新点 1: 从第 2 轮起, 用上一轮模型重新计算难例权重并重建采样器
        if cfg.use_hard_sampling and epoch > 1:
            print(f'\n  [难例采样] 第 {epoch} 轮: 用上一轮模型对训练集重新打分并更新采样权重...')
            sampler, weights = hard_sampler.refresh(model, train_dataset, cfg)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=cfg.num_workers)
            info = hard_sampler.log_epoch(epoch - 1, weights)
            print(f'    权重均值 {info["mean"]} / 最大 {info["max"]} / '
                  f'被明显加权的难例占比 {info["hard_sample_ratio"] * 100:.1f}%')

        loss, cost = train_one_epoch(model, criterion, optimizer, scheduler,
                                     train_loader, cfg, epoch, cfg.epochs)

        # 每轮在 dev 上评估(Micro-F1 作为早停依据)
        probs, y_true, _ = run_inference(model, dev_loader, cfg)
        metrics = compute_metrics(y_true, probs, cfg.class_list, threshold=cfg.label_threshold)
        print(f'  -> epoch {epoch}: train_loss={loss:.4f} | dev Micro-F1={metrics["micro_f1"]:.4f} '
              f'| dev Macro-F1={metrics["macro_f1"]:.4f} | 耗时 {cost:.1f}s')

        history.append({
            'epoch': epoch, 'train_loss': round(loss, 4),
            'dev_micro_f1': metrics['micro_f1'], 'dev_macro_f1': metrics['macro_f1'],
            'dev_subset_accuracy': metrics['subset_accuracy'], 'seconds': round(cost, 1),
        })

        # 保存最优 + 早停
        if metrics['micro_f1'] > best_micro_f1:
            best_micro_f1, best_epoch, no_improve = metrics['micro_f1'], epoch, 0
            model.save(cfg.model_save_path, extra={
                'best_epoch': epoch,
                'dev_micro_f1': best_micro_f1,
                'pos_weight': pos_weight.tolist(),
            })
            print(f'     [保存] dev Micro-F1 提升至 {best_micro_f1:.4f}, 已保存最优模型')
        else:
            no_improve += 1
            print(f'     [早停计数] 第 {no_improve}/{cfg.patience} 轮未提升')
            if no_improve >= cfg.patience:
                print(f'  [早停] 连续 {cfg.patience} 轮未提升, 提前结束训练')
                break

        with open(cfg.train_log_path, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg.summary().split('\n'), 'history': history},
                      f, ensure_ascii=False, indent=2)

    print(f'\n[5/6] 训练结束: 最优 epoch={best_epoch}, dev Micro-F1={best_micro_f1:.4f}')

    # todo 7. 概率校准 + 双阈值标定 + 测试集最终评估
    print('\n[6/6] 加载最优模型, 在 dev 上校准概率并标定阈值...')
    from multilabel_model import load_trained_model
    best_model, meta = load_trained_model(cfg, cfg.model_save_path)

    # ---- 6.1 概率校准: 只在 dev 上拟合, test 不参与 ----
    # 与 02-rf / 03-fasttext 用同一套工具, 保证三套模型的"置信度"可比。
    # BERT 的 sigmoid 输出同样会偏自信, 不校准的话对照表里的"平均置信度"又是鸡同鸭讲。
    dev_probs, dev_true, _ = run_inference(best_model, dev_loader, cfg)
    dev_probs = np.asarray(_to_np(dev_probs))
    dev_true = np.asarray(_to_np(dev_true))
    calibration = fit_calibrators(dev_probs, dev_true, cfg.class_list)
    dev_probs_cal = apply_calibrators(dev_probs, calibration)

    # ---- 6.2 双阈值标定(业务约束: 拒识率 <= target_reject_rate) ----
    print(f'  在 dev 上网格搜索双阈值(目标拒识率 <= {cfg.target_reject_rate:.0%}) ...')
    grid = grid_search_thresholds(dev_probs_cal, dev_true, cfg.class_list,
                                  target_reject_rate=cfg.target_reject_rate)
    if grid:
        top = grid[0]
        cfg.label_threshold = top['label_threshold']
        cfg.global_threshold = top['global_threshold']
        print(f"  候选组合 {len(grid)} 个, 最优: 单标签阈值 {top['label_threshold']} / "
              f"全局阈值 {top['global_threshold']}")
        print(f"  -> 自动分流率 {top['auto_rate']:.2%}, 拒识率 {top['reject_rate']:.2%}, "
              f"自动分流 Micro-F1 {top['auto_micro_f1']}")
    else:
        print('  [警告] 没有组合能满足拒识率上限, 保持默认阈值 '
              f'({cfg.label_threshold}/{cfg.global_threshold})')

    # ---- 6.3 把校准器与标定后的阈值写回模型产物 ----
    # 推理端(bert_predict_fun)会从 meta 里读这两样, 保证线上线下同一口径
    best_model.save(cfg.model_save_path, extra={
        'calibration': calibration,
        'tuned_on': 'dev',
        'tuned_target_reject_rate': cfg.target_reject_rate,
    })
    print(f'  校准器与标定阈值已写回 -> {cfg.model_save_path}')

    # ---- 6.4 测试集最终评估(用校准后的概率) ----
    if test_loader is not None:
        probs, y_true, _ = run_inference(best_model, test_loader, cfg)
        probs = np.asarray(_to_np(probs))
        y_true = np.asarray(_to_np(y_true))
        probs_cal = apply_calibrators(probs, calibration)
        print(calibration_summary(probs, probs_cal, y_true, calibration,
                                  '04-bert 概率校准效果(在 test 上评估)'))
        test_metrics = compute_metrics(y_true, probs_cal, cfg.class_list,
                                       threshold=cfg.label_threshold)
        test_metrics['threshold'] = cfg.label_threshold
        from model2dev_utils import compute_business_metrics
        test_metrics.update(compute_business_metrics(probs_cal, y_true, cfg, cfg.class_list))
        print_metrics(test_metrics, title='测试集最终指标')
        save_metrics(test_metrics, os.path.join(cfg.result_dir, 'test_metrics.json'))
    else:
        test_metrics = None
        print('  [提示] 未找到 test.txt, 跳过测试集评估')

    print('\n' + '=' * 72)
    print('训练完成 [OK]')
    print(f'  最优模型 : {cfg.model_save_path}')
    print(f'  训练日志 : {cfg.train_log_path}')
    if test_metrics:
        print(f'  测试指标 : Micro-F1={test_metrics["micro_f1"]:.4f}  '
              f'Macro-F1={test_metrics["macro_f1"]:.4f}  '
              f'拒识率={test_metrics.get("reject_rate", 0):.4f}')
    print('  下一步   : python 04-bert/bert_predict_fun.py   # 单条推理自测')
    print('=' * 72)


if __name__ == '__main__':
    main()