import random
import time
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def now_ms():
    return int(time.perf_counter() * 1000)
