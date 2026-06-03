#!/usr/bin/env python3
"""
RAR/ZIP 密码恢复工具 - GPU 加速版本
架构：
- GPU 模式：调用 Hashcat（百万级密码/秒）或 GPU-CPU 混合流水线
- CPU 模式：多线程优化（万级密码/秒）

GPU 加速原理：
ZIP/RAR 密码验证包含 AES/ZipCrypto 加密运算，纯 Python 无法直接在 GPU 上执行。
本脚本提供两种 GPU 方案：
1. Hashcat 集成：提取文件哈希，调用 Hashcat 的 CUDA 内核（推荐，速度最快）
2. GPU-CPU 混合流水线：GPU 批量生成候选密码，CPU 验证（无需额外工具）
"""

import argparse
import itertools
import string
import time
import sys
import os
import subprocess
import tempfile
import struct
import hashlib
import threading
import multiprocessing
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ZIP/RAR 库
import zipfile
try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False

# GPU 库
try:
    import cupy as cp
    GPU_CUPY_AVAILABLE = True
except ImportError:
    GPU_CUPY_AVAILABLE = False


# ============================================================
# 模块 1: Hashcat GPU 破解（最高性能）
# ============================================================

def find_hashcat() -> Optional[str]:
    """查找 Hashcat 可执行文件"""
    candidates = [
        'hashcat',
        'hashcat.exe',
        'hashcat64.bin',
        os.path.expanduser('~/.local/bin/hashcat'),
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # 尝试常见安装路径
    paths = [
        r'C:\Program Files\hashcat\hashcat.exe',
        r'C:\hashcat\hashcat.exe',
    ]
    for path in paths:
        if os.path.exists(path):
            return path

    return None


def extract_zip_hash(filepath: str) -> Optional[str]:
    """提取 ZIP 文件的 John/Hashcat 格式哈希"""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()

        # 查找 local file header signature (0x04034b50)
        # 简化实现：尝试使用 zip2john 格式
        # 实际更复杂，这里提供基础支持

        # 方法：提取加密文件信息
        with zipfile.ZipFile(filepath, 'r') as zf:
            for info in zf.infolist():
                if info.file_size > 0 and info.compress_size > 0:
                    # 检查是否加密
                    if info.flag_bits & 0x1:
                        # 加密文件，返回文件路径供 hashcat 直接处理
                        return filepath
        return None
    except Exception:
        return None


def crack_with_hashcat(
    filepath: str,
    file_type: str,
    charset: str,
    min_len: int,
    max_len: int,
    hashcat_path: str,
    attack_mode: str = 'bruteforce'
) -> Optional[str]:
    """使用 Hashcat 进行 GPU 加速破解"""
    # Hashcat 模式号
    # 13600 = WinRAR
    # 17220 = PDF
    # 20500 = PKZIP (encrypted)
    # 14700 = RAR3
    if file_type == 'zip':
        mode = '20500'  # PKZIP
    elif file_type == 'rar':
        mode = '13600'  # WinRAR
    else:
        print(f"[!] 不支持的文件类型: {file_type}")
        return None

    # 构建 Hashcat 命令
    cmd = [
        hashcat_path,
        '-m', mode,
        '-a', '3',  # 掩码攻击（暴力破解）
        '-w', '3',  # 工作负载：高
        '--force',
    ]

    # 字符集和长度
    if attack_mode == 'bruteforce':
        # 构建掩码
        mask = f'?{get_charset_mask(charset)}'
        if min_len != max_len:
            # 多长度：从 min_len 到 max_len
            mask = f'{"?" * min_len}' if min_len < max_len else mask
            cmd.extend(['--increment', '--increment-min', str(min_len)])

        cmd.extend(['-1', charset, mask])

    elif attack_mode == 'dictionary':
        # 字典模式需要字典文件参数，由调用方处理
        pass

    cmd.append(filepath)

    print(f"[*] 执行 Hashcat: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None  # 无超时
        )
        # 解析输出
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip().startswith('Session..........: hashcat'):
                    pass
                # Hashcat 会在 cracked 时输出密码
            # 尝试从 potfile 读取
            potfile = os.path.expanduser('~/.hashcat/hashcat.potfile')
            if os.path.exists(potfile):
                with open(potfile, 'r') as f:
                    for line in f:
                        if filepath in line or os.path.basename(filepath) in line:
                            parts = line.rsplit(':', 1)
                            if len(parts) == 2:
                                return parts[1].strip()
        return None
    except subprocess.TimeoutExpired:
        print("[!] Hashcat 超时")
        return None
    except Exception as e:
        print(f"[!] Hashcat 执行失败: {e}")
        return None


def get_charset_mask(charset: str) -> str:
    """将字符集转换为 Hashcat 掩码字符"""
    mask = ''
    if any(c in string.digits for c in charset):
        mask += 'd'
    if any(c in string.ascii_lowercase for c in charset):
        mask += 'l'
    if any(c in string.ascii_uppercase for c in charset):
        mask += 'u'
    if any(c in string.punctuation for c in charset):
        mask += 's'
    return mask or 'a'  # 默认 all


# ============================================================
# 模块 2: GPU-CPU 混合流水线（无需 Hashcat）
# ============================================================

class GPUPasswordGenerator:
    """
    GPU 加速密码生成器
    使用 CuPy 在 GPU 上批量生成候选密码，大幅减少 CPU 负担
    """

    def __init__(self, charset: str, min_len: int, max_len: int, batch_size: int = 100000):
        self.charset = charset
        self.charset_array = cp.array([ord(c) for c in charset], dtype=cp.int32)
        self.charset_size = len(charset)
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.current_length = min_len
        self.current_index = 0

    def _index_to_password_gpu(self, indices: cp.ndarray) -> list:
        """将索引数组批量转换为密码列表（GPU 加速）"""
        passwords = []
        length = self.current_length

        # 使用 GPU 批量计算每个位置的字符
        for i in range(length):
            # indices // (charset_size ^ (length - 1 - i)) % charset_size
            divisor = self.charset_size ** (length - 1 - i)
            char_indices = (indices // divisor) % self.charset_size
            # GPU 查表
            chars = self.charset_array[char_indices]
            passwords.append(chars)

        # 转置并转换回 CPU
        if passwords:
            password_matrix = cp.stack(passwords, axis=1)
            # 批量转换为字符串
            cpu_matrix = cp.asnumpy(password_matrix)
            return [
                ''.join(chr(c) for c in row)
                for row in cpu_matrix
            ]
        return []

    def next_batch(self) -> list:
        """获取下一批密码（GPU 生成）"""
        if self.current_length > self.max_len:
            return []

        total_for_length = self.charset_size ** self.current_length
        remaining = total_for_length - self.current_index
        count = min(self.batch_size, remaining)

        if count <= 0:
            self.current_length += 1
            self.current_index = 0
            return self.next_batch()

        # 生成索引范围（GPU）
        start_idx = self.current_index
        indices = cp.arange(start_idx, start_idx + count, dtype=cp.int64)

        # GPU 批量转换
        passwords = self._index_to_password_gpu(indices)

        self.current_index += count
        if self.current_index >= total_for_length:
            self.current_length += 1
            self.current_index = 0

        return passwords


class HybridPipeline:
    """
    GPU-CPU 混合流水线架构
    - GPU 线程：批量生成候选密码
    - CPU 线程池：验证密码
    - 无锁队列连接两者
    """

    def __init__(self, filepath: str, file_type: str, tester, generator, workers: int):
        self.filepath = filepath
        self.file_type = file_type
        self.tester = tester
        self.generator = generator
        self.workers = workers
        self.queue = None
        self.tested = 0
        self.found = False
        self.result_password = None
        self.start_time = 0
        self.last_progress_time = 0
        self.lock = threading.Lock()

    def _gpu_producer(self):
        """GPU 线程：批量生成密码放入队列"""
        batch_size = 5000

        if hasattr(self.generator, 'next_batch'):
            # GPU 生成器
            while not self.found:
                batch = self.generator.next_batch()
                if not batch:
                    break
                self.queue.put(batch)
        else:
            # 普通生成器：批量读取
            batch = []
            for pwd in self.generator:
                batch.append(pwd)
                if len(batch) >= batch_size:
                    self.queue.put(batch)
                    batch = []
            if batch:
                self.queue.put(batch)

        # 发送结束信号
        self.queue.put(None)

    def _cpu_worker(self, worker_id: int):
        """CPU 工作线程：从队列取密码并验证"""
        local_zf = None
        if hasattr(self.tester, '_get_local_zf'):
            local_zf = self.tester._get_local_zf()

        while True:
            if self.found:
                break

            batch = self.queue.get()
            if batch is None:
                self.queue.put(None)  # 转发结束信号
                break

            for pwd in batch:
                if self.found:
                    break
                if self.tester.test(pwd):
                    with self.lock:
                        if not self.found:
                            self.found = True
                            self.result_password = pwd
                    break
                with self.lock:
                    self.tested += 1

    def _update_progress(self):
        now = time.time()
        if now - self.last_progress_time >= 0.5:
            elapsed = now - self.start_time
            speed = self.tested / elapsed if elapsed > 0 else 0
            print(
                f"\r[*] 已测试: {self.tested:,} | "
                f"速度: {speed:,.0f} 密码/秒 | "
                f"用时: {elapsed:.1f}s",
                end='',
                flush=True
            )
            self.last_progress_time = now

    def run(self) -> Optional[str]:
        """执行混合流水线"""
        import queue
        self.queue = queue.Queue(maxsize=10)  # 缓冲 10 批
        self.start_time = time.time()
        self.last_progress_time = time.time()

        # 启动 GPU 生产者
        producer = threading.Thread(target=self._gpu_producer, daemon=True)
        producer.start()

        # 启动 CPU 工作线程
        cpu_threads = []
        for i in range(self.workers):
            t = threading.Thread(target=self._cpu_worker, args=(i,), daemon=True)
            t.start()
            cpu_threads.append(t)

        # 监控进度
        progress_interval = 0.5
        while producer.is_alive() or not self.queue.empty():
            self._update_progress()
            if self.found:
                break
            time.sleep(progress_interval)

        producer.join()
        for t in cpu_threads:
            t.join()

        return self.result_password if self.found else None

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        speed = self.tested / elapsed if elapsed > 0 else 0
        return {
            'tested': self.tested,
            'elapsed': elapsed,
            'speed': speed
        }


# ============================================================
# 模块 3: 高性能 CPU 测试器（复用句柄）
# ============================================================

class ZipTester:
    """ZIP 密码测试器 - 每个线程独立文件句柄"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._local = threading.local()
        with zipfile.ZipFile(filepath, 'r') as zf:
            self.test_name = zf.namelist()[0] if zf.namelist() else None

    def _get_local_zf(self):
        if not hasattr(self._local, 'zf') or self._local.zf is None:
            self._local.zf = zipfile.ZipFile(self.filepath, 'r')
        return self._local.zf

    def test(self, password: str) -> bool:
        try:
            zf = self._get_local_zf()
            zf.read(self.test_name, pwd=password.encode('utf-8', errors='ignore'))
            return True
        except (RuntimeError, zipfile.BadZipFile):
            return False
        except Exception:
            return False

    def close(self):
        if hasattr(self._local, 'zf') and self._local.zf:
            self._local.zf.close()


class RarTester:
    """RAR 密码测试器 - 每个线程独立文件句柄"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._local = threading.local()
        with rarfile.RarFile(filepath, 'r') as rf:
            self.test_name = rf.namelist()[0] if rf.namelist() else None

    def _get_local_rf(self):
        if not hasattr(self._local, 'rf') or self._local.rf is None:
            self._local.rf = rarfile.RarFile(self.filepath, 'r')
        return self._local.rf

    def test(self, password: str) -> bool:
        try:
            rf = self._get_local_rf()
            with rf.open(self.test_name, pwd=password) as f:
                f.read(1)
            return True
        except (rarfile.BadRarFile, rarfile.NeedFirstVolume, rarfile.Error):
            return False
        except Exception:
            return False

    def close(self):
        if hasattr(self._local, 'rf') and self._local.rf:
            self._local.rf.close()


# ============================================================
# 模块 4: CPU 破解引擎
# ============================================================

class CPU_Cracker:
    def __init__(self, tester, generator, workers):
        self.tester = tester
        self.generator = generator
        self.workers = workers
        self.tested = 0
        self.found = False
        self.result_password = None
        self.start_time = 0
        self.last_progress_time = 0
        self.lock = threading.Lock()

    def _worker(self, batch):
        for pwd in batch:
            if self.found:
                return None
            if self.tester.test(pwd):
                with self.lock:
                    if not self.found:
                        self.found = True
                        self.result_password = pwd
                return pwd
            with self.lock:
                self.tested += 1
        return None

    def crack(self, batch_size=500) -> Optional[str]:
        self.start_time = time.time()
        self.last_progress_time = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            batch = []
            futures = {}

            for pwd in self.generator:
                batch.append(pwd)

                if len(batch) >= batch_size:
                    fut = executor.submit(self._worker, batch)
                    futures[fut] = batch
                    batch = []

                    # 检查完成的
                    done = [f for f in futures if f.done()]
                    for fut in done:
                        if fut.result():
                            for f in futures:
                                if f != fut and not f.done():
                                    f.cancel()
                            return fut.result()
                        del futures[fut]

                    self._progress(pwd)

            if batch:
                fut = executor.submit(self._worker, batch)
                futures[fut] = batch

            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    return result
                self._progress('')

        return None

    def _progress(self, pwd):
        now = time.time()
        if now - self.last_progress_time >= 0.5:
            elapsed = now - self.start_time
            speed = self.tested / elapsed if elapsed > 0 else 0
            print(
                f"\r[*] 已测试: {self.tested:,} | "
                f"速度: {speed:,.0f} 密码/秒 | "
                f"用时: {elapsed:.1f}s | "
                f"当前: {pwd}",
                end='',
                flush=True
            )
            self.last_progress_time = now

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            'tested': self.tested,
            'elapsed': elapsed,
            'speed': self.tested / elapsed if elapsed > 0 else 0
        }


# ============================================================
# 生成器
# ============================================================

def bruteforce_gen(charset_tuple, min_len, max_len):
    for length in range(min_len, max_len + 1):
        for combo in itertools.product(charset_tuple, repeat=length):
            yield ''.join(combo)


def dict_gen(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd:
                yield pwd


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='RAR/ZIP 密码恢复工具（GPU 加速版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU 加速模式：
  --gpu hashcat   调用 Hashcat（最快，需安装 Hashcat）
  --gpu pipeline  GPU-CPU 混合流水线（需 CuPy）
  --gpu auto      自动选择（默认）

示例：
  # Hashcat GPU 暴力破解
  python password_cracker.py -f archive.zip -t zip --gpu hashcat -c 0123456789 --min 1 --max 6

  # GPU-CPU 混合流水线
  python password_cracker.py -f archive.zip -t zip --gpu pipeline -c 0123456789 --min 1 --max 6

  # 纯 CPU 多线程
  python password_cracker.py -f archive.zip -t zip --cpu -w 8
        """
    )

    parser.add_argument('-f', '--file', required=True, help='目标文件')
    parser.add_argument('-t', '--type', choices=['zip', 'rar'], required=True, help='文件类型')
    parser.add_argument('-m', '--mode', choices=['bruteforce', 'dictionary'], default='bruteforce', help='破解模式')
    parser.add_argument('-c', '--charset', default=None, help='字符集')
    parser.add_argument('--min', type=int, default=1, help='最小密码长度')
    parser.add_argument('--max', type=int, default=8, help='最大密码长度')
    parser.add_argument('-d', '--dictionary', default=None, help='字典文件')
    parser.add_argument('-w', '--workers', type=int, default=None, help='CPU 线程数')
    parser.add_argument('--gpu', choices=['auto', 'hashcat', 'pipeline'], default='auto',
                        help='GPU 模式（默认 auto）')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--hashcat-path', default=None, help='Hashcat 路径')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] 文件不存在: {args.file}")
        sys.exit(1)

    if args.mode == 'dictionary' and not args.dictionary:
        print("[!] 字典模式需要 -d 参数")
        sys.exit(1)

    # 检测 GPU 能力
    use_gpu = not args.cpu
    gpu_backend = None

    if use_gpu:
        # 1. 检查 Hashcat
        hashcat_path = args.hashcat_path or find_hashcat()
        if hashcat_path and args.gpu in ('auto', 'hashcat'):
            gpu_backend = 'hashcat'
            print(f"[*] GPU 加速: Hashcat ({hashcat_path})")
        # 2. 检查 CuPy
        elif GPU_CUPY_AVAILABLE and args.gpu in ('auto', 'pipeline'):
            gpu_backend = 'pipeline'
            print(f"[*] GPU 加速: GPU-CPU 混合流水线 (CuPy)")
        else:
            print("[*] 未检测到 GPU 后端，使用 CPU 模式")

    workers = args.workers or multiprocessing.cpu_count()

    print(f"\n{'='*60}")
    print(f"[*] 文件: {args.file}")
    print(f"[*] 类型: {args.type}")
    print(f"[*] 模式: {args.mode}")
    print(f"[*] 后端: {gpu_backend or 'CPU'} ({workers} 线程)")
    if args.mode == 'bruteforce':
        charset = args.charset or (string.ascii_letters + string.digits)
        print(f"[*] 字符集: {charset}")
        print(f"[*] 长度: {args.min} - {args.max}")
    print(f"{'='*60}\n")

    # 初始化测试器
    if args.type == 'zip':
        tester = ZipTester(args.file)
    else:
        if not RAR_AVAILABLE:
            print("[!] 需要安装 rarfile: pip install rarfile")
            sys.exit(1)
        tester = RarTester(args.file)

    # 初始化生成器
    charset = args.charset or (string.ascii_letters + string.digits)
    if args.mode == 'bruteforce':
        charset_tuple = tuple(charset)
        if gpu_backend == 'pipeline' and GPU_CUPY_AVAILABLE:
            generator = GPUPasswordGenerator(charset, args.min, args.max, batch_size=50000)
        else:
            generator = bruteforce_gen(charset_tuple, args.min, args.max)
    else:
        generator = dict_gen(args.dictionary)

    # 执行破解
    password = None

    if gpu_backend == 'hashcat':
        hashcat_path = args.hashcat_path or find_hashcat()
        print(f"[*] 启动 Hashcat GPU 破解...")
        password = crack_with_hashcat(
            args.file, args.type, charset,
            args.min, args.max, hashcat_path,
            args.mode
        )
    elif gpu_backend == 'pipeline':
        print(f"[*] 启动 GPU-CPU 混合流水线...")
        pipeline = HybridPipeline(args.file, args.type, tester, generator, workers)
        password = pipeline.run()
        stats = pipeline.get_stats()
    else:
        print(f"[*] 启动 CPU 多线程破解...")
        cracker = CPU_Cracker(tester, generator, workers)
        password = cracker.crack()
        stats = cracker.get_stats()

    tester.close()

    # 输出结果
    print(f"\n\n{'='*60}")
    if password:
        print(f"[+] 密码找到: {password}")
        if gpu_backend != 'hashcat':
            print(f"[+] 测试数: {stats['tested']:,}")
            print(f"[+] 速度: {stats['speed']:,.0f} 密码/秒")
            print(f"[+] 用时: {stats['elapsed']:.2f} 秒")
    else:
        print(f"[-] 未找到密码")
        if gpu_backend != 'hashcat':
            print(f"[-] 测试数: {stats['tested']:,}")
            print(f"[-] 用时: {stats['elapsed']:.2f} 秒")
    print(f"{'='*60}")

    sys.exit(0 if password else 1)


if __name__ == '__main__':
    main()
