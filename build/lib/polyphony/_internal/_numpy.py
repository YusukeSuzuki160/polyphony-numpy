from .typing import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, complex64, complex128, bool_, List, Tuple
from .timing import clkfence, clksleep


def array_8_complex128(a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, ) -> Tuple[int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64]:
	return a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, 


def fft_1_8_complex128(a_real_0: int64, a_imag_0: int64, a_real_1: int64, a_imag_1: int64, a_real_2: int64, a_imag_2: int64, a_real_3: int64, a_imag_3: int64, a_real_4: int64, a_imag_4: int64, a_real_5: int64, a_imag_5: int64, a_real_6: int64, a_imag_6: int64, a_real_7: int64, a_imag_7: int64, ) -> Tuple[int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64]:
	t0_real_0, t0_imag_0 = complex128_add(a_real_0, a_imag_0, a_real_4, a_imag_4)
	t0_real_1, t0_imag_1 = complex128_sub(a_real_0, a_imag_0, a_real_4, a_imag_4)
	t0_real_2, t0_imag_2 = complex128_add(a_real_1, a_imag_1, a_real_5, a_imag_5)
	t0_real_3, t0_imag_3 = complex128_sub(a_real_1, a_imag_1, a_real_5, a_imag_5)
	t0_real_4, t0_imag_4 = complex128_add(a_real_2, a_imag_2, a_real_6, a_imag_6)
	t0_real_5, t0_imag_5 = complex128_sub(a_real_2, a_imag_2, a_real_6, a_imag_6)
	t0_real_6, t0_imag_6 = complex128_add(a_real_3, a_imag_3, a_real_7, a_imag_7)
	t0_real_7, t0_imag_7 = complex128_sub(a_real_3, a_imag_3, a_real_7, a_imag_7)
	clkfence()
	w0_real, w0_imag = complex128_mult(t0_real_4, t0_imag_4, 65536, 0)
	t1_real_0, t1_imag_0 = complex128_add(t0_real_0, t0_imag_0, w0_real, w0_imag)
	t1_real_1, t1_imag_1 = complex128_sub(t0_real_0, t0_imag_0, w0_real, w0_imag)
	w2_real, w2_imag = complex128_mult(t0_real_5, t0_imag_5, 0, -65536)
	t1_real_2, t1_imag_2 = complex128_add(t0_real_1, t0_imag_1, w2_real, w2_imag)
	t1_real_3, t1_imag_3 = complex128_sub(t0_real_1, t0_imag_1, w2_real, w2_imag)
	w4_real, w4_imag = complex128_mult(t0_real_6, t0_imag_6, 65536, 0)
	t1_real_4, t1_imag_4 = complex128_add(t0_real_2, t0_imag_2, w4_real, w4_imag)
	t1_real_5, t1_imag_5 = complex128_sub(t0_real_2, t0_imag_2, w4_real, w4_imag)
	w6_real, w6_imag = complex128_mult(t0_real_7, t0_imag_7, 0, -65536)
	t1_real_6, t1_imag_6 = complex128_add(t0_real_3, t0_imag_3, w6_real, w6_imag)
	t1_real_7, t1_imag_7 = complex128_sub(t0_real_3, t0_imag_3, w6_real, w6_imag)
	clkfence()
	w0_real, w0_imag = complex128_mult(t1_real_4, t1_imag_4, 65536, 0)
	t2_real_0, t2_imag_0 = complex128_add(t1_real_0, t1_imag_0, w0_real, w0_imag)
	t2_real_1, t2_imag_1 = complex128_sub(t1_real_0, t1_imag_0, w0_real, w0_imag)
	w2_real, w2_imag = complex128_mult(t1_real_5, t1_imag_5, 0, -65536)
	t2_real_2, t2_imag_2 = complex128_add(t1_real_1, t1_imag_1, w2_real, w2_imag)
	t2_real_3, t2_imag_3 = complex128_sub(t1_real_1, t1_imag_1, w2_real, w2_imag)
	w4_real, w4_imag = complex128_mult(t1_real_6, t1_imag_6, 46340, -46340)
	t2_real_4, t2_imag_4 = complex128_add(t1_real_2, t1_imag_2, w4_real, w4_imag)
	t2_real_5, t2_imag_5 = complex128_sub(t1_real_2, t1_imag_2, w4_real, w4_imag)
	w6_real, w6_imag = complex128_mult(t1_real_7, t1_imag_7, -46340, -46340)
	t2_real_6, t2_imag_6 = complex128_add(t1_real_3, t1_imag_3, w6_real, w6_imag)
	t2_real_7, t2_imag_7 = complex128_sub(t1_real_3, t1_imag_3, w6_real, w6_imag)
	clkfence()
	t_real_0 = t2_real_0
	t_imag_0 = t2_imag_0
	t_real_4 = t2_real_1
	t_imag_4 = t2_imag_1
	t_real_2 = t2_real_2
	t_imag_2 = t2_imag_2
	t_real_6 = t2_real_3
	t_imag_6 = t2_imag_3
	t_real_1 = t2_real_4
	t_imag_1 = t2_imag_4
	t_real_5 = t2_real_5
	t_imag_5 = t2_imag_5
	t_real_3 = t2_real_6
	t_imag_3 = t2_imag_6
	t_real_7 = t2_real_7
	t_imag_7 = t2_imag_7
	clkfence()
	return t_real_0, t_imag_0, t_real_1, t_imag_1, t_real_2, t_imag_2, t_real_3, t_imag_3, t_real_4, t_imag_4, t_real_5, t_imag_5, t_real_6, t_imag_6, t_real_7, t_imag_7, 


def complex128_add(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    c_real = a_real + b_real
    c_imag = a_imag + b_imag
    return c_real, c_imag


def complex128_sub(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    c_real = a_real - b_real
    c_imag = a_imag - b_imag
    return c_real, c_imag


def complex128_mult(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    real = (a_real * b_real - a_imag * b_imag) >> 16
    imag = (a_real * b_imag + a_imag * b_real) >> 16
    return real, imag


def _print_8_complex128(a_0: int64, a_1: int64, a_2: int64, a_3: int64, a_4: int64, a_5: int64, a_6: int64, a_7: int64, a_8: int64, a_9: int64, a_10: int64, a_11: int64, a_12: int64, a_13: int64, a_14: int64, a_15: int64, ) -> None:
	print(a_0)
	print(a_1)
	print(a_2)
	print(a_3)
	print(a_4)
	print(a_5)
	print(a_6)
	print(a_7)
	print(a_8)
	print(a_9)
	print(a_10)
	print(a_11)
	print(a_12)
	print(a_13)
	print(a_14)
	print(a_15)



