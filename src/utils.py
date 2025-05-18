import sys
import time
from functools import wraps


def logfile_enabled(prefix: str):
    """
    Redirects the print statements to a log file.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")

            # Backup the original stdout
            ori_stdout = sys.stdout

            try:
                with open(f"{prefix}-{timestr}.log", "a+") as f:
                    sys.stdout = f
                    return func(*args, **kwargs)
            finally:
                # Restore the original stdout
                sys.stdout = ori_stdout

        return wrapper

    return decorator
