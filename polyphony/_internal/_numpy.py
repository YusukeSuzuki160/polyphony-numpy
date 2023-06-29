from polyphony.typing import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, complex64, complex128, bool_, List, Tuple

def array(a, dtype=None):
    pass

def mul(a_00: int64, a_01: int64, a_10: int64, a_11: int64, b_00: int64, b_01: int64, b_10: int64, b_11: int64) -> Tuple[int64, int64, int64, int64]:
    c_00: int64 = a_00 * b_00 + a_01 * b_10
    c_01: int64 = a_00 * b_01 + a_01 * b_11
    c_10: int64 = a_10 * b_00 + a_11 * b_10
    c_11: int64 = a_10 * b_01 + a_11 * b_11

    return c_00, c_01, c_10, c_11

