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
    def __init__(self, a, dtype=None):
        self.mem = a
        self.dtype = dtype

def array(a, dtype=None) -> Ndarray:
    return Ndarray(a, dtype)
