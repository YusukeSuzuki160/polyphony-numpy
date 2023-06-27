class int8:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype int8'


class int16:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype int16'


class int32:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype int32'


class int64:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype int64'
    

class uint8:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype uint8'


class uint16:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype uint16'


class uint32:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype uint32'


class uint64:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype uint64'


class float16:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype float16'


class float32:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype float32'


class float64:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype float64'
    

class complex64:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype complex64'
    
    
class complex128:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype complex128'
    

class bool_:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'dtype bool'
    

class Ndarray:
    def __init__(self, a, dtype=None, shape=None):
        self.mem = a
        self.dtype = dtype
        self.shape = (len(a), len(a[0]))
def array(a, dtype=None) -> Ndarray:
    return Ndarray(a, dtype)

def zeros(shape, dtype=None) -> Ndarray:
    if dtype == int8 or dtype == int16 or dtype == int32 or dtype == int64:
        zero = 0
    elif dtype == uint8 or dtype == uint16 or dtype == uint32 or dtype == uint64:
        zero = 0
    elif dtype == float16 or dtype == float32 or dtype == float64:
        zero = 0.0
    elif dtype == complex64 or dtype == complex128:
        zero = 0.0 + 0.0j
    elif dtype == bool_:
        zero = False
    re_list = [[zero] for i in range(shape[0] * shape[1])]
    return Ndarray(re_list, dtype)

def add(a, b):
    ret = zeros(a.shape, a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ret.mem[i * a.shape[1] + j] = a.mem[i * a.shape[1] + j] + b.mem[i * a.shape[1] + j]
    return ret

def sub(a, b):
    ret = zeros(a.shape, a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ret.mem[i * a.shape[1] + j] = a.mem[i * a.shape[1] + j] - b.mem[i * a.shape[1] + j]
    return ret

def mul(a, b):
    ret = zeros(a.shape, a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[1]):
                ret.mem[i * a.shape[1] + j] += a.mem[i * a.shape[1] + k] * b.mem[k * a.shape[1] + j]
    return ret

