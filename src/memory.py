import ctypes
import gc

import psutil
import torch
from tabulate import tabulate


def print_memory_stats():
    gb = 1 << 30
    stats = psutil.virtual_memory()

    data = [
        ["Total", f"{stats.total / gb:.2f} GB"],
        ["Available", f"{stats.available / gb:.2f} GB"],
        ["Used", f"{stats.used / gb:.2f} GB"],
        ["Free", f"{stats.free / gb:.2f} GB"],
        ["Percent Used", f"{stats.percent:.1f}%"],
    ]

    mem_table = tabulate(
        data,
        headers=["Metric", "Value"],
        tablefmt="psql",
    )

    print(mem_table)


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
