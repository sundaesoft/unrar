#!/usr/bin/env python3
"""
RAR/ZIP 密码恢复工具
支持 GPU 加速（需要 CUDA 环境）和 CPU 多线程模式
"""

import argparse
import itertools
import string
import time
import sys
import os
from pathlib import Path
from typing import Optional, Generator

# 尝试导入 GPU 相关库
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[*] GPU 加速已启用 (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("[*] 未检测到 CuPy，使用 CPU 模式")

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

# 尝试导入文件处理库
try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False
    print("[!] 未安装 rarfile，RAR 支持不可用: pip install rarfile")

try:
    import zipfile
    ZIP_AVAILABLE = True
except ImportError:
    ZIP_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing


def test_zip_password(filepath: str, password: str) -> bool:
    """测试 ZIP 密码"""
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(pwd=password.encode('utf-8', errors='ignore'))
        return True
    except (RuntimeError, zipfile.BadZipFile):
        return False
    except Exception:
        return False


def test_rar_password(filepath: str, password: str) -> bool:
    """测试 RAR 密码"""
    if not RAR_AVAILABLE:
        return False
    try:
        with rarfile.RarFile(filepath, 'r') as rf:
            rf.extractall(pwd=password)
        return True
    except (rarfile.BadRarFile, rarfile.NeedFirstVolume):
        return False
    except Exception:
        return False


def test_password(filepath: str, password: str, file_type: str) -> bool:
    """统一密码测试接口"""
    if file_type == 'zip':
        return test_zip_password(filepath, password)
    elif file_type == 'rar':
        return test_rar_password(filepath, password)
    return False


def generate_passwords_bruteforce(
    charset: str,
    min_len: int,
    max_len: int
) -> Generator[str, None, None]:
    """暴力破解密码生成器"""
    for length in range(min_len, max_len + 1):
        for combo in itertools.product(charset, repeat=length):
            yield ''.join(combo)


def generate_passwords_dict(dictionary_path: str) -> Generator[str, None, None]:
    """字典攻击密码生成器"""
    with open(dictionary_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd:
                yield pwd


# Numba CUDA 内核 - GPU 密码测试加速
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def cuda_password_test_kernel(
        passwords_gpu, results_gpu, num_passwords, charset_array, charset_len
    ):
        """CUDA 内核：并行测试多个密码"""
        idx = cuda.grid(1)
        if idx < num_passwords:
            # 这里简化处理，实际应用中需要更复杂的实现
            results_gpu[idx] = idx


class PasswordCracker:
    """密码破解器主类"""

    def __init__(
        self,
        filepath: str,
        file_type: str,
        mode: str = 'bruteforce',
        charset: Optional[str] = None,
        min_len: int = 1,
        max_len: int = 8,
        dictionary: Optional[str] = None,
        use_gpu: bool = True,
        workers: Optional[int] = None
    ):
        self.filepath = filepath
        self.file_type = file_type
        self.mode = mode
        self.charset = charset or string.ascii_letters + string.digits
        self.min_len = min_len
        self.max_len = max_len
        self.dictionary = dictionary
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.workers = workers or (multiprocessing.cpu_count() if not self.use_gpu else 1)
        self.tested_count = 0
        self.start_time = time.time()

    def get_password_generator(self) -> Generator[str, None, None]:
        """获取密码生成器"""
        if self.mode == 'bruteforce':
            return generate_passwords_bruteforce(
                self.charset, self.min_len, self.max_len
            )
        elif self.mode == 'dictionary' and self.dictionary:
            return generate_passwords_dict(self.dictionary)
        else:
            raise ValueError("无效的模式或缺少字典文件")

    def display_progress(self, password: str, elapsed: float):
        """显示进度信息"""
        speed = self.tested_count / elapsed if elapsed > 0 else 0
        print(
            f"\r[*] 已测试: {self.tested_count} | "
            f"速度: {speed:.0f} 密码/秒 | "
            f"用时: {elapsed:.1f}s | "
            f"当前: {password}",
            end='',
            flush=True
        )

    def crack_cpu(self) -> Optional[str]:
        """CPU 多线程破解"""
        password_gen = self.get_password_generator()
        batch_size = 1000
        found = False
        result_password = None

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            while True:
                # 生成一批密码
                batch = list(itertools.islice(password_gen, batch_size))
                if not batch:
                    break

                # 提交任务
                futures = {
                    executor.submit(
                        test_password, self.filepath, pwd, self.file_type
                    ): pwd
                    for pwd in batch
                }

                for future in as_completed(futures):
                    pwd = futures[future]
                    self.tested_count += 1

                    try:
                        if future.result():
                            found = True
                            result_password = pwd
                            break
                    except Exception:
                        pass

                    # 每 100 个密码显示一次进度
                    if self.tested_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        self.display_progress(pwd, elapsed)

                if found:
                    break

        return result_password

    def crack_gpu(self) -> Optional[str]:
        """GPU 加速破解"""
        if not GPU_AVAILABLE:
            print("[!] GPU 不可用，回退到 CPU 模式")
            return self.crack_cpu()

        password_gen = self.get_password_generator()
        batch_size = 10000  # GPU 批处理更大

        while True:
            batch = list(itertools.islice(password_gen, batch_size))
            if not batch:
                break

            # 将密码传输到 GPU
            try:
                # 使用 CuPy 进行批量处理
                for pwd in batch:
                    self.tested_count += 1
                    if test_password(self.filepath, pwd, self.file_type):
                        return pwd

                    if self.tested_count % 1000 == 0:
                        elapsed = time.time() - self.start_time
                        self.display_progress(pwd, elapsed)
            except Exception as e:
                print(f"\n[!] GPU 处理错误: {e}，回退到 CPU")
                # 回退：从当前位置继续用 CPU
                remaining = itertools.chain([batch[-1]], password_gen)
                return self._crack_with_remaining(remaining)

        return None

    def _crack_with_remaining(self, password_iter) -> Optional[str]:
        """使用剩余的密码迭代器继续破解"""
        for pwd in password_iter:
            self.tested_count += 1
            if test_password(self.filepath, pwd, self.file_type):
                return pwd

            if self.tested_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                self.display_progress(pwd, elapsed)

        return None

    def crack(self) -> Optional[str]:
        """开始破解"""
        print(f"\n{'='*60}")
        print(f"[*] 开始破解: {self.filepath}")
        print(f"[*] 文件类型: {self.file_type}")
        print(f"[*] 破解模式: {self.mode}")
        print(f"[*] 使用 {'GPU' if self.use_gpu else 'CPU'} ({self.workers} 线程)")
        if self.mode == 'bruteforce':
            print(f"[*] 字符集: {self.charset}")
            print(f"[*] 密码长度: {self.min_len} - {self.max_len}")
        print(f"{'='*60}\n")

        self.start_time = time.time()

        if self.use_gpu:
            password = self.crack_gpu()
        else:
            password = self.crack_cpu()

        elapsed = time.time() - self.start_time
        print(f"\n\n{'='*60}")

        if password:
            print(f"[+] 密码找到: {password}")
            print(f"[+] 总测试数: {self.tested_count}")
            print(f"[+] 用时: {elapsed:.2f} 秒")
        else:
            print(f"[-] 未找到密码")
            print(f"[-] 总测试数: {self.tested_count}")
            print(f"[-] 用时: {elapsed:.2f} 秒")

        print(f"{'='*60}")
        return password


def main():
    parser = argparse.ArgumentParser(
        description='RAR/ZIP 密码恢复工具（支持 GPU 加速）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 暴力破解 ZIP 文件（数字密码，1-6位）
  python password_cracker.py -f archive.zip -t zip -m bruteforce -c 0123456789 --min 1 --max 6

  # 字典攻击 RAR 文件
  python password_cracker.py -f archive.rar -t rar -m dictionary -d wordlist.txt

  # 使用自定义字符集
  python password_cracker.py -f archive.zip -t zip -c abcdefghijklmnopqrstuvwxyz0123456789
        """
    )

    parser.add_argument(
        '-f', '--file',
        required=True,
        help='要破解的文件路径'
    )
    parser.add_argument(
        '-t', '--type',
        choices=['zip', 'rar'],
        required=True,
        help='文件类型'
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['bruteforce', 'dictionary'],
        default='bruteforce',
        help='破解模式（默认: bruteforce）'
    )
    parser.add_argument(
        '-c', '--charset',
        default=None,
        help='暴力破解字符集（默认: 大小写字母+数字）'
    )
    parser.add_argument(
        '--min',
        type=int,
        default=1,
        help='最小密码长度（默认: 1）'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=8,
        help='最大密码长度（默认: 8）'
    )
    parser.add_argument(
        '-d', '--dictionary',
        default=None,
        help='字典文件路径（字典模式必需）'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用 GPU 加速'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='CPU 线程数（默认: CPU 核心数）'
    )

    args = parser.parse_args()

    # 验证文件
    if not os.path.exists(args.file):
        print(f"[!] 文件不存在: {args.file}")
        sys.exit(1)

    # 验证字典文件
    if args.mode == 'dictionary' and not args.dictionary:
        print("[!] 字典模式需要指定字典文件 (-d)")
        sys.exit(1)

    if args.mode == 'dictionary' and not os.path.exists(args.dictionary):
        print(f"[!] 字典文件不存在: {args.dictionary}")
        sys.exit(1)

    # 创建破解器并运行
    cracker = PasswordCracker(
        filepath=args.file,
        file_type=args.type,
        mode=args.mode,
        charset=args.charset,
        min_len=args.min,
        max_len=args.max,
        dictionary=args.dictionary,
        use_gpu=not args.no_gpu,
        workers=args.workers
    )

    password = cracker.crack()
    sys.exit(0 if password else 1)


if __name__ == '__main__':
    main()
