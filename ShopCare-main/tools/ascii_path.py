"""
ASCII 路径桥接 (tools/ascii_path.py)

要解决的问题:
    fasttext 的 C++ 层用 std::ifstream / ofstream 打开文件, 收到的是 UTF-8 字节串;
    Windows 上遇到非 ASCII 路径(典型: 用户名带中文, 如 C:/Users/吴贵/...)会按系统
    ANSI 代码页去解释这串字节, 于是打开失败, 报出来的却是:
        ValueError: ... cannot be opened for training!
    而 Python 自己的 open()/os.path.exists() 走宽字符 API, **同一个路径完全正常**.
    症状因此非常迷惑: "文件明明写出来了, fasttext 却说打不开".
    训练(input=语料、save_model)与推理(load_model)三个环节都会中招.

做法:
    * 路径纯 ASCII -> 什么都不做, 原样返回(绝大多数机器走这条分支, 零开销).
    * 含非 ASCII -> 把文件复制到"确定是 ASCII 的工作目录", 只让 fasttext 看见 ASCII 路径;
      写方向的产物再复制回真实目标. 复制用的是 Python 的宽字符 IO, 因此必然成功.

为什么不干脆要求"项目必须放在纯英文路径":
    那是把环境问题推给用户, 而且队友换个机器就要重新踩. 桥接是本地无感的, 留着更安全.
"""

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager

# 候选工作目录, 按优先级排列. 只有"路径本身是 ASCII 且真的可写"的才会被采用.
_WORKDIR_CANDIDATES = (
    os.environ.get('SHOPCARE_ASCII_WORKDIR'),   # 留个口子: CI/服务器可显式指定
    'C:/Windows/Temp',
    'C:/Users/Public',
    '/tmp',
    tempfile.gettempdir(),
)

_WORKDIR = None


def needs_bridge(path):
    """路径里含非 ASCII 字符时, fasttext 就打不开它"""
    return not str(path).isascii()


def workspace():
    """返回一个 ASCII + 可写的公共工作目录(懒探测 + 进程内缓存)"""
    global _WORKDIR
    if _WORKDIR and os.path.isdir(_WORKDIR):
        return _WORKDIR
    tried = []
    for base in _WORKDIR_CANDIDATES:
        if not base:
            continue
        base = str(base)
        if not base.isascii():
            tried.append(f'{base}(路径含非 ASCII)')
            continue
        target = os.path.join(base, 'shopcare_ascii')
        try:
            os.makedirs(target, exist_ok=True)
            probe = os.path.join(target, '.write_test')
            with open(probe, 'w') as fh:
                fh.write('ok')
            os.remove(probe)
        except OSError as exc:
            tried.append(f'{target}({type(exc).__name__})')
            continue
        _WORKDIR = target
        return target
    raise RuntimeError('找不到可用的 ASCII 临时目录, 已尝试: ' + '; '.join(tried))


def _stage_path(path, tag):
    """给真实路径算一个稳定且 ASCII 的暂存路径

    稳定很重要: 语料要在 train/eval 多次调用之间复用同一份副本, 不能每次都重拷 20MB.
    """
    digest = hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()[:10]
    name = os.path.basename(str(path))
    # 文件名本身也可能是中文, 只保留 ASCII 的扩展名部分
    ext = os.path.splitext(name)[1]
    if not ext.isascii():
        ext = ''
    return os.path.join(workspace(), f'{tag}-{digest}{ext}')


def _sync(src, dst):
    """src 比 dst 新(或 dst 不存在)时才复制, 避免重复 IO"""
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
            and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    shutil.copyfile(src, dst)
    return dst


@contextmanager
def readable(path):
    """以 fasttext 能打开的形式读取 path"""
    path = str(path)
    if not needs_bridge(path):
        yield path
        return
    staged = _stage_path(path, 'in')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    yield _sync(path, staged)


@contextmanager
def writable(path):
    """让 fasttext 写到暂存文件, 退出上下文时再复制回 path(含中文的真实路径)"""
    path = str(path)
    if not needs_bridge(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        yield path
        return
    staged = _stage_path(path, 'out')
    yield staged
    if not os.path.exists(staged):
        raise RuntimeError(f'fasttext 没有产出文件, 暂存路径: {staged}')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    shutil.copyfile(staged, path)


if __name__ == '__main__':
    print('=' * 72)
    print('ascii_path 自测')
    print('=' * 72)
    print('工作目录:', workspace())
    assert workspace().isascii()
    print('中文路径判定:', needs_bridge('C:/Users/吴贵/a.txt'),
          '| 英文路径判定:', needs_bridge('C:/Users/public/a.txt'))
    assert needs_bridge('C:/Users/吴贵/a.txt') and not needs_bridge('C:/a.txt')

    real = os.path.join(workspace(), 'demo_中文_目录', 'm.bin')
    os.makedirs(os.path.dirname(real), exist_ok=True)
    with writable(real) as p:
        with open(p, 'wb') as fh:
            fh.write(b'\x01\x02\x03')
    with readable(real) as p:
        assert not needs_bridge(p), '交给 fasttext 的路径仍含非 ASCII'
        with open(p, 'rb') as fh:
            assert fh.read() == b'\x01\x02\x03'
    shutil.rmtree(os.path.dirname(real), ignore_errors=True)
    print('\n[OK] ascii_path 自测通过(读/写两个方向都能桥接)')
