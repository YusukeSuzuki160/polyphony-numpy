from polyphony.typing import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, complex64, complex128, bool_, List, Tuple
from polyphony.timing import clkfence, clksleep


def array_4_complex128(a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, ) -> Tuple[int64, int64, int64, int64, int64, int64, int64, int64]:
	return a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, 


def fft_1_4_complex128(a_real_0: int64, a_imag_0: int64, a_real_1: int64, a_imag_1: int64, a_real_2: int64, a_imag_2: int64, a_real_3: int64, a_imag_3: int64, ) -> Tuple[int64, int64, int64, int64, int64, int64, int64, int64]:
	t_real_0, t_imag_0 = complex128_add(a_real_0, a_imag_0, a_real_2, a_imag_2)
	t_real_1, t_imag_1 = complex128_sub(a_real_0, a_imag_0, a_real_2, a_imag_2)
	clksleep(1)
	t_real_2, t_imag_2 = complex128_add(a_real_1, a_imag_1, a_real_3, a_imag_3)
	t_real_3, t_imag_3 = complex128_sub(a_real_1, a_imag_1, a_real_3, a_imag_3)
	clksleep(1)
	w_real, w_imag = complex128_mult(t_real_2, t_imag_2, 65536, 0)
	t_real = t_real_0
	t_imag = t_imag_0
	t_real_0, t_imag_0 = complex128_add(t_real, t_imag, w_real, w_imag)
	t_real_2, t_imag_2 = complex128_sub(t_real, t_imag, w_real, w_imag)
	clksleep(1)
	w_real, w_imag = complex128_mult(t_real_3, t_imag_3, 0, 65536)
	t_real = t_real_1
	t_imag = t_imag_1
	t_real_1, t_imag_1 = complex128_add(t_real, t_imag, w_real, w_imag)
	t_real_3, t_imag_3 = complex128_sub(t_real, t_imag, w_real, w_imag)
	clksleep(1)
	return t_real_0, t_imag_0, t_real_1, t_imag_1, t_real_2, t_imag_2, t_real_3, t_imag_3, 


def complex128_add(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    return a_real + b_real, a_imag + b_imag


def complex128_sub(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    return a_real - b_real, a_imag - b_imag


def complex128_mult(a_real: int64, a_imag: int64, b_real: int64, b_imag: int64) -> Tuple[int64, int64]:
    real = (a_real * b_real - a_imag * b_imag) >> 16
    imag = (a_real * b_imag + a_imag * b_real) >> 16
    return real, imag


def _print_4_complex128(a_0: int64, a_1: int64, a_2: int64, a_3: int64, a_4: int64, a_5: int64, a_6: int64, a_7: int64, ) -> None:
	print(a_0)
	print(a_1)
	print(a_2)
	print(a_3)
	print(a_4)
	print(a_5)
	print(a_6)
	print(a_7)



