import ctypes
import gc
import platform

import psutil
import torch
from tabulate import tabulate


def print_system_specs():
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print("CUDA Available:", is_cuda_available)
    # Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_cuda_devices)
    if is_cuda_available:
        for i in range(num_cuda_devices):
            # Get CUDA device properties
            device = torch.device("cuda", i)
            print(f"--- CUDA Device {i} ---")
            print("Name:", torch.cuda.get_device_name(i))
            print("Compute Capability:", torch.cuda.get_device_capability(i))
            print(
                "Total Memory:",
                torch.cuda.get_device_properties(i).total_memory,
                "bytes",
            )
    # Get CPU information
    print("--- CPU Information ---")
    print("Processor:", platform.processor())
    print("System:", platform.system(), platform.release())
    print("Python Version:", platform.python_version())


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
