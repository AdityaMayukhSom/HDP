import ctypes
import gc

import psutil
import torch


def print_memory_stats():
    gb = 2**30
    stats = psutil.virtual_memory()
    free_gb = stats.free / gb
    print(f"Your runtime has {free_gb:.1f} gigabytes of free RAM")
    used_gb = stats.used / gb
    print(f"Your runtime has {used_gb:.1f} gigabytes of used RAM")
    avlb_gb = stats.available / gb
    print(f"Your runtime has {avlb_gb:.1f} gigabytes of available RAM")
    ram_gb = stats.total / gb
    print(f"Your runtime has {ram_gb:.1f} gigabytes of total RAM")
    print(f"Your runtime has {stats.percent:.1f}% usage of RAM")


def empty_all_memory():
    """
    Forcefully releases unused memory from both GPU and CPU.

    - Frees GPU memory by clearing the CUDA cache (`torch.cuda.empty_cache()`).
    - Triggers Python's garbage collector to release unreferenced objects. Hence
      it's useful to delete object references before calling this.
    - Calls `malloc_trim(0)` via `libc` to return freed heap memory back to the OS (Linux only).
    """
    libc = ctypes.CDLL("libc.so.6")
    torch.cuda.empty_cache()
    gc.collect()
    libc.malloc_trim(0)
