from time import time
import logging
import sys

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func



def setup_logger(name, log_file, level=logging.INFO, stdout = False):
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')

    """To setup as many loggers as you want"""
    if stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger