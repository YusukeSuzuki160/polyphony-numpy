from enum import IntEnum, auto


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
    
    
    def __repr__(self):
        return self.name


    
    
