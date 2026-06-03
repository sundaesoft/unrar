#!/usr/bin/env python3
"""
RAR/ZIP 密码恢复工具 - GPU 加速版本
支持三种模式：
1. Hashcat GPU（最快，需安装 Hashcat + zip2john/john）
2. GPU-CPU 混合流水线（需 CuPy）
3. CPU 多线程优化
"""

import argparse
import itertools
import string
import time
import sys
import os
import subprocess
import tempfile
import threading
import multiprocessing
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import zipfile
try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False

try:
    import cupy as cp
    GPU_CUPY_AVAILABLE = True
except ImportError:
    GPU_CUPY_AVAILABLE = False


# ==================== Hashcat 相关 ====================

def find_hashcat() -> Optional[str]:
    candidates = ['hashcat', 'hashcat.exe']
    for cmd in candidates:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except:
            continue
    for path in [r'C:\Program Files\hashcat\hashcat.exe', r'C:\hashcat\hashcat.exe']:
        if os.path.exists(path):
            return path
    return None


def find_zip2john() -> Optional[str]:
    candidates = ['zip2john', 'zip2john.py', '/usr/bin/zip2john', '/usr/share/john/zip2john']
    for cmd in candidates:
        try:
            if os.path.exists(cmd):
                return cmd
        except:
            continue
    return None


def extract_zip_hash(filepath: str) -> Optional[str]:
    """用 zip2john 提取 ZIP 哈希"""
    zip2john = find_zip2john()
    if not zip2john:
        return None
    try:
        result = subprocess.run([zip2john, filepath], capture_output=True, text=True, timeout=30)
        if result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        print(f"[!] zip2john 失败：{e}")
    return None


def crack_with_hashcat(filepath: str, file_type: str, charset: str, min_len: int, max_len: int, hashcat_path: str) -> Optional[str]:
    """使用 Hashcat 破解"""
    # 模式：17600=WinZip AES, 13600=WinRAR
    mode = '17600' if file_type == 'zip' else '13600'
    
    # ZIP 需要先用 zip2john 提取哈希
    hash_input = filepath
    temp_hash_file = None
    
    if file_type == 'zip':
        hash_data = extract_zip_hash(filepath)
        if hash_data:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.hash') as f:
                f.write(hash_data)
                temp_hash_file = f.name
                hash_input = f.name
                print(f"[*] 已提取 ZIP 哈希")
    
    # 构建掩码字符
    mask_char = ''
    if any(c in string.digits for c in charset): mask_char += 'd'
    if any(c in string.ascii_lowercase for c in charset): mask_char += 'l'
    if any(c in string.ascii_uppercase for c in charset): mask_char += 'u'
    if any(c in string.punctuation for c in charset): mask_char += 's'
    mask_char = mask_char or 'a'
    
    # 构建命令
    cmd = [hashcat_path, '-m', mode, '-a', '3', '-w', '3', '--force']
    
    if min_len == max_len:
        mask = mask_char * min_len
    else:
        mask = mask_char * min_len
        cmd.extend(['--increment', '--increment-max', str(max_len)])
    
    cmd.extend(['-1', charset, mask, hash_input])
    
    print(f"[*] Hashcat 命令：{' '.join(cmd)}")
    print(f"[*] 模式：{mode} | 掩码：{mask}")
    
    try:
        # 直接运行，显示完整输出
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        
        # 从 potfile 读取密码
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
        print(f"[!] Hashcat 失败：{e}")
        return None
    finally:
        if temp_hash_file and os.path.exists(temp_hash_file):
            os.unlink(temp_hash_file)


# ==================== GPU-CPU 混合流水线 ====================

class GPUPasswordGenerator:
    def __init__(self, charset: str, min_len: int, max_len: int, batch_size: int = 50000):
        self.charset = charset
        self.charset_array = cp.array([ord(c) for c in charset], dtype=cp.int32)
        self.charset_size = len(charset)
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.current_length = min_len
        self.current_index = 0

    def next_batch(self) -> list:
        if self.current_length > self.max_len:
            return []
        
        total = self.charset_size ** self.current_length
        remaining = total - self.current_index
        count = min(self.batch_size, remaining)
        
        if count <= 0:
            self.current_length += 1
            self.current_index = 0
            return self.next_batch()
        
        indices = cp.arange(self.current_index, self.current_index + count, dtype=cp.int64)
        passwords = self._indices_to_passwords(indices)
        
        self.current_index += count
        if self.current_index >= total:
            self.current_length += 1
            self.current_index = 0
        
        return passwords

    def _indices_to_passwords(self, indices: cp.ndarray) -> list:
        length = self.current_length
        chars_list = []
        
        for i in range(length):
            divisor = self.charset_size ** (length - 1 - i)
            char_indices = (indices // divisor) % self.charset_size
            chars = self.charset_array[char_indices]
            chars_list.append(chars)
        
        if chars_list:
            matrix = cp.stack(chars_list, axis=1)
            cpu_matrix = cp.asnumpy(matrix)
            return [''.join(chr(c) for c in row) for row in cpu_matrix]
        return []


class HybridPipeline:
    def __init__(self, tester, generator, workers: int):
        self.tester = tester
        self.generator = generator
        self.workers = workers
        self.queue = None
        self.tested = 0
        self.found = False
        self.result = None
        self.start_time = 0
        self.last_progress = 0
        self.lock = threading.Lock()

    def _producer(self):
        batch_size = 5000
        if hasattr(self.generator, 'next_batch'):
            while not self.found:
                batch = self.generator.next_batch()
                if not batch:
                    break
                self.queue.put(batch)
        else:
            batch = []
            for pwd in self.generator:
                batch.append(pwd)
                if len(batch) >= batch_size:
                    self.queue.put(batch)
                    batch = []
            if batch:
                self.queue.put(batch)
        self.queue.put(None)

    def _worker(self, wid: int):
        while not self.found:
            batch = self.queue.get()
            if batch is None:
                self.queue.put(None)
                break
            for pwd in batch:
                if self.found:
                    break
                if self.tester.test(pwd):
                    with self.lock:
                        if not self.found:
                            self.found = True
                            self.result = pwd
                    break
                with self.lock:
                    self.tested += 1

    def run(self) -> Optional[str]:
        import queue
        self.queue = queue.Queue(maxsize=10)
        self.start_time = time.time()
        self.last_progress = time.time()
        
        producer = threading.Thread(target=self._producer, daemon=True)
        producer.start()
        
        threads = []
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        
        while producer.is_alive() or not self.queue.empty():
            if self.found:
                break
            now = time.time()
            if now - self.last_progress >= 0.5:
                elapsed = now - self.start_time
                speed = self.tested / elapsed if elapsed > 0 else 0
                print(f"\r[*] 已测试：{self.tested:,} | 速度：{speed:,.0f}/秒 | 用时：{elapsed:.1f}s", end='', flush=True)
                self.last_progress = now
            time.sleep(0.5)
        
        producer.join()
        for t in threads:
            t.join()
        
        return self.result

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {'tested': self.tested, 'elapsed': elapsed, 'speed': self.tested / elapsed if elapsed > 0 else 0}


# ==================== CPU 测试器 ====================

class ZipTester:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._local = threading.local()
        with zipfile.ZipFile(filepath, 'r') as zf:
            self.test_name = zf.namelist()[0] if zf.namelist() else None

    def _get_zf(self):
        if not hasattr(self._local, 'zf') or self._local.zf is None:
            self._local.zf = zipfile.ZipFile(self.filepath, 'r')
        return self._local.zf

    def test(self, password: str) -> bool:
        try:
            zf = self._get_zf()
            zf.read(self.test_name, pwd=password.encode('utf-8', errors='ignore'))
            return True
        except:
            return False

    def close(self):
        if hasattr(self._local, 'zf') and self._local.zf:
            self._local.zf.close()


class RarTester:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._local = threading.local()
        with rarfile.RarFile(filepath, 'r') as rf:
            self.test_name = rf.namelist()[0] if rf.namelist() else None

    def _get_rf(self):
        if not hasattr(self._local, 'rf') or self._local.rf is None:
            self._local.rf = rarfile.RarFile(self.filepath, 'r')
        return self._local.rf

    def test(self, password: str) -> bool:
        try:
            rf = self._get_rf()
            with rf.open(self.test_name, pwd=password) as f:
                f.read(1)
            return True
        except:
            return False

    def close(self):
        if hasattr(self._local, 'rf') and self._local.rf:
            self._local.rf.close()


# ==================== CPU 破解引擎 ====================

class CPU_Cracker:
    def __init__(self, tester, generator, workers):
        self.tester = tester
        self.generator = generator
        self.workers = workers
        self.tested = 0
        self.found = False
        self.result = None
        self.start_time = 0
        self.last_progress = 0
        self.lock = threading.Lock()

    def _worker(self, batch):
        for pwd in batch:
            if self.found:
                return None
            if self.tester.test(pwd):
                with self.lock:
                    if not self.found:
                        self.found = True
                        self.result = pwd
                return pwd
            with self.lock:
                self.tested += 1
        return None

    def crack(self, batch_size=500) -> Optional[str]:
        self.start_time = time.time()
        self.last_progress = time.time()
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            batch = []
            futures = {}
            
            for pwd in self.generator:
                batch.append(pwd)
                if len(batch) >= batch_size:
                    fut = executor.submit(self._worker, batch)
                    futures[fut] = batch
                    batch = []
                    
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
        if now - self.last_progress >= 0.5:
            elapsed = now - self.start_time
            speed = self.tested / elapsed if elapsed > 0 else 0
            print(f"\r[*] 已测试：{self.tested:,} | 速度：{speed:,.0f}/秒 | 用时：{elapsed:.1f}s | 当前：{pwd}", end='', flush=True)
            self.last_progress = now

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {'tested': self.tested, 'elapsed': elapsed, 'speed': self.tested / elapsed if elapsed > 0 else 0}


# ==================== 生成器 ====================

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


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='RAR/ZIP 密码恢复工具（GPU 加速）')
    parser.add_argument('-f', '--file', required=True, help='目标文件')
    parser.add_argument('-t', '--type', choices=['zip', 'rar'], required=True, help='文件类型')
    parser.add_argument('-m', '--mode', choices=['bruteforce', 'dictionary'], default='bruteforce', help='破解模式')
    parser.add_argument('-c', '--charset', default=None, help='字符集')
    parser.add_argument('--min', type=int, default=1, help='最小长度')
    parser.add_argument('--max', type=int, default=8, help='最大长度')
    parser.add_argument('-d', '--dictionary', help='字典文件')
    parser.add_argument('-w', '--workers', type=int, default=None, help='CPU 线程数')
    parser.add_argument('--gpu', choices=['auto', 'hashcat', 'pipeline'], default='auto', help='GPU 模式')
    parser.add_argument('--cpu', action='store_true', help='强制 CPU')
    parser.add_argument('--hashcat-path', help='Hashcat 路径')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] 文件不存在：{args.file}")
        sys.exit(1)

    if args.mode == 'dictionary' and not args.dictionary:
        print("[!] 字典模式需要 -d 参数")
        sys.exit(1)

    # 检测 GPU
    use_gpu = not args.cpu
    gpu_backend = None

    if use_gpu:
        hashcat_path = args.hashcat_path or find_hashcat()
        if hashcat_path and args.gpu in ('auto', 'hashcat'):
            gpu_backend = 'hashcat'
            print(f"[*] GPU 加速：Hashcat")
        elif GPU_CUPY_AVAILABLE and args.gpu in ('auto', 'pipeline'):
            gpu_backend = 'pipeline'
            print(f"[*] GPU 加速：GPU-CPU 混合流水线")
        else:
            print("[*] 未检测到 GPU 后端，使用 CPU")

    workers = args.workers or multiprocessing.cpu_count()

    charset = args.charset or (string.ascii_letters + string.digits)

    print(f"\n{'='*60}")
    print(f"[*] 文件：{args.file}")
    print(f"[*] 类型：{args.type}")
    print(f"[*] 模式：{args.mode}")
    print(f"[*] 后端：{gpu_backend or 'CPU'} ({workers} 线程)")
    if args.mode == 'bruteforce':
        print(f"[*] 字符集：{charset}")
        print(f"[*] 长度：{args.min} - {args.max}")
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
    stats = None

    if gpu_backend == 'hashcat':
        hashcat_path = args.hashcat_path or find_hashcat()
        print(f"[*] 启动 Hashcat...")
        password = crack_with_hashcat(args.file, args.type, charset, args.min, args.max, hashcat_path)
    elif gpu_backend == 'pipeline':
        print(f"[*] 启动 GPU-CPU 流水线...")
        pipeline = HybridPipeline(tester, generator, workers)
        password = pipeline.run()
        stats = pipeline.get_stats()
    else:
        print(f"[*] 启动 CPU 多线程...")
        cracker = CPU_Cracker(tester, generator, workers)
        password = cracker.crack()
        stats = cracker.get_stats()

    tester.close()

    print(f"\n\n{'='*60}")
    if password:
        print(f"[+] 密码找到：{password}")
        if stats:
            print(f"[+] 测试数：{stats['tested']:,}")
            print(f"[+] 速度：{stats['speed']:,.0f} 密码/秒")
            print(f"[+] 用时：{stats['elapsed']:.2f} 秒")
    else:
        print(f"[-] 未找到密码")
        if stats:
            print(f"[-] 测试数：{stats['tested']:,}")
            print(f"[-] 用时：{stats['elapsed']:.2f} 秒")
    print(f"{'='*60}")

    sys.exit(0 if password else 1)


if __name__ == '__main__':
    main()
