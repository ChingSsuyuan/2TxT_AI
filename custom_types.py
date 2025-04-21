import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import sys
import platform
from typing import Tuple, List, Union, Callable, Type, Iterator, Dict, Set, Optional, Any, Sized
from enum import Enum

# 检测操作系统
IS_WINDOWS = sys.platform == 'win32'
IS_MAC = platform.system() == 'Darwin'

get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None

# 基本类型定义保持不变
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

# 设备处理修改
D = torch.device
CPU = torch.device('cpu')

def get_device(device_id: int) -> D:
    # 在Mac上默认使用CPU
    if IS_MAC:
        return CPU
    
    # 对于其他系统，检查CUDA可用性
    if not torch.cuda.is_available():
        return CPU
    
    # 确保device_id有效
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

# 为了保持兼容性，保留CUDA别名
CUDA = get_device