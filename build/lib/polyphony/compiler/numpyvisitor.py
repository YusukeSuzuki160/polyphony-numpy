from typing import List, Dict, Tuple, Union, Optional
from enum import IntEnum, auto
import ast

class Dtype(IntEnum):
    Int8 = auto()
    Int16 = auto()
    Int32 = auto()
    Int64 = auto()
    UInt8 = auto()
    UInt16 = auto()
    UInt32 = auto()
    UInt64 = auto()
    Float16 = auto()
    Float32 = auto()
    Float64 = auto()
    Float128 = auto()
    Bool = auto()
    Complex64 = auto()
    Complex128 = auto()
    Complex256 = auto()
    Unicode = auto()
    Object = auto()  # TODO: support object type
    
    @staticmethod
    def default_int(self):
        return Dtype.Int64
    
    @staticmethod
    def default_float(self):
        return Dtype.Float64
    
    @staticmethod
    def default_complex(self):
        return Dtype.Complex128
    
    
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name


class NumpyTable:
    def __init__(self, shape: Tuple[int, ...], dtype: Dtype):
        self.shape = shape
        self.dtype = dtype
    
    def __str__(self):
        return 'NumpyTable(shape={}, dtype={})'.format(self.shape, self.dtype)
    
    
