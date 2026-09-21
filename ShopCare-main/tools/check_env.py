"""
ShopCare 环境自检 (tools)

用途:
    这个项目横跨三个技术栈(RF/sklearn、FastText、BERT/torch、FastAPI 服务),
    依赖不少。跑任何一个阶段之前先执行本脚本, 能立刻看清**缺什么、装什么、影响了哪个阶段**,
    而不是等到训练到一半才报 ModuleNotFoundError。

用法:
    python tools/check_env.py           # 检查全部
    python tools/check_env.py --stage rf
    python tools/check_env.py --stage backend

说明:
    * 本项目**不自动联网下载任何模型**; 本脚本也不会联网, 只做 import 探测。
    * 缺依赖时给出的是"装什么", 不会替你执行 pip install。
"""

import argparse
import importlib
import json
import os
import platform
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (import 名, pip 包名, 版本属性, 属于哪个阶段, 是否必需)
PACKAGES = [
    ('numpy', 'numpy', '__version__', 'all', True),
    ('sklearn', 'scikit-learn', '__version__', 'rf', True),
    ('scipy', 'scipy', '__version__', 'rf', True),
    ('joblib', 'joblib', '__version__', 'rf', True),
    ('jieba', 'jieba', '__version__', 'rf/fasttext', False),
    ('fasttext', 'fasttext', '__version__', 'fasttext', True),
    ('pandas', 'pandas', '__version__', 'analysis', False),
    ('torch', 'torch', '__version__', 'bert', True),
    ('transformers', 'transformers', '__version__', 'bert', True),
    ('fastapi', 'fastapi', '__version__', 'backend', True),
    ('uvicorn', 'uvicorn', '__version__', 'backend', True),
    ('pydantic', 'pydantic', '__version__', 'backend', True),
    ('pymysql', 'PyMySQL', '__version__', 'backend', True),
    ('redis', 'redis', '__version__', 'backend', True),
    ('jwt', 'PyJWT', '?', 'backend', True),
    ('passlib', 'passlib', '__version__', 'backend', False),
    ('httpx', 'httpx', '__version__', 'tools/llm', False),
]

STAGE_ALIASES = {
    'all': 'all', 'rf': 'rf', '02': 'rf', 'fasttext': 'fasttext', 'ft': 'fasttext', '03': 'fasttext',
    'bert': 'bert', '04': 'bert', 'backend': 'backend', 'api': 'backend', 'analysis': 'analysis',
}


def probe(module_name, attr):
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, attr, '?') if attr != '?' else '?'
        return True, str(ver)
    except ImportError as exc:
        return False, type(exc).__name__
    except Exception as exc:                     # noqa: BLE001  装了但 import 就崩(版本不兼容/缺 DLL)
        return False, f'{type(exc).__name__}: {exc}'


def main():
    parser = argparse.ArgumentParser(description='ShopCare 环境自检')
    parser.add_argument('--stage', default='all',
                        help='只检查某个阶段: all / rf / fasttext / bert / backend / analysis')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出(供脚本消费)')
    args = parser.parse_args()
    stage = STAGE_ALIASES.get(args.stage.lower(), args.stage.lower())

    rows, missing_required = [], []
    for mod, pkg, attr, belongs, required in PACKAGES:
        if stage != 'all' and belongs not in (stage, 'all'):
            continue
        ok, ver = probe(mod, attr)
        rows.append({'module': mod, 'pip': pkg, 'stage': belongs,
                     'required': required, 'installed': ok, 'version': ver})
        if not ok and required:
            missing_required.append(pkg)

    info = {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': f'{platform.system()} {platform.release()}',
        'cwd': os.getcwd(),
        'project_root': PROJECT_ROOT,
        'stage_filter': stage,
        'packages': rows,
        'missing_required': missing_required,
    }

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if not missing_required else 1

    print('=' * 78)
    print('ShopCare 环境自检')
    print('=' * 78)
    print(f'  Python      : {info["python"]}')
    print(f'  解释器      : {info["executable"]}')
    print(f'  系统        : {info["platform"]}')
    print(f'  过滤阶段    : {stage}')
    print('-' * 78)
    print(f'  {"模块":<16}{"pip 包名":<18}{"阶段":<12}{"必需":<6}{"状态"}')
    print('-' * 78)
    for r in rows:
        status = ('已安装 ' + r['version']) if r['installed'] else '缺失 (' + r['version'] + ')'
        print(f'  {r["module"]:<16}{r["pip"]:<18}{r["stage"]:<12}'
              f'{"是" if r["required"] else "否":<6}{status}')

    print('-' * 78)
    if missing_required:
        print('[警告] 必需依赖缺失, 对应阶段无法运行:')
        print('       pip install ' + ' '.join(missing_required))
    else:
        print('[OK] 所选阶段的必需依赖齐全')

    # 数据文件顺带检查一下, 免得装完环境才发现没数据
    print('-' * 78)
    print('  数据文件:')
    for name in ('class.txt', 'train.txt', 'dev.txt', 'test.txt', 'label_meta.json'):
        p = os.path.join(PROJECT_ROOT, '01-data', name)
        size = f'{os.path.getsize(p) / 1024:.0f} KB' if os.path.exists(p) else '缺失'
        print(f'    {name:<18}{size}')

    model_checks = [
        ('02-rf', os.path.join(PROJECT_ROOT, '02-rf', 'save_models', 'rf_tfidf.joblib')),
        ('03-fasttext', os.path.join(PROJECT_ROOT, '03-fasttext', 'save_models', 'fasttext_multilabel.bin')),
        ('04-bert', os.path.join(PROJECT_ROOT, '04-bert', 'save_models', 'bert_lora_multilabel.pt')),
        ('04-bert 预训练模型(需含 config.json + 权重)', os.path.join(PROJECT_ROOT, '04-bert', 'bert-base-chinese')),
    ]
    print('  模型产物:')
    for name, p in model_checks:
        # '目录存在' 不等于 '模型放好了': 空的 bert-base-chinese 目录必须报成未就绪,
        # 否则用户会以为可以训练, 跑起来才在 from_pretrained 处失败
        ok = os.path.exists(p)
        if ok and p.endswith('bert-base-chinese'):
            ok = os.path.exists(os.path.join(p, 'config.json')) and (
                os.path.exists(os.path.join(p, 'pytorch_model.bin'))
                or os.path.exists(os.path.join(p, 'model.safetensors')))
        print(f'    {name:<22}{"已就绪" if ok else "未生成/未放置"}  ({p})')

    return 0 if not missing_required else 1


if __name__ == '__main__':
    sys.exit(main())