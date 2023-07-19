from .typing import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, complex64, complex128, bool_, List, Tuple
from .timing import clkfence, clksleep
from . import module, is_worker_running
from .io import Port

class NpFFT:
	def __init__(self):
		self.a_0_real = Port(int64, 'in', protocol='valid')
		self.a_0_imag = Port(int64, 'in', protocol='valid')
		self.a_1_real = Port(int64, 'in', protocol='valid')
		self.a_1_imag = Port(int64, 'in', protocol='valid')
		self.a_2_real = Port(int64, 'in', protocol='valid')
		self.a_2_imag = Port(int64, 'in', protocol='valid')
		self.a_3_real = Port(int64, 'in', protocol='valid')
		self.a_3_imag = Port(int64, 'in', protocol='valid')
		self.a_4_real = Port(int64, 'in', protocol='valid')
		self.a_4_imag = Port(int64, 'in', protocol='valid')
		self.a_5_real = Port(int64, 'in', protocol='valid')
		self.a_5_imag = Port(int64, 'in', protocol='valid')
		self.a_6_real = Port(int64, 'in', protocol='valid')
		self.a_6_imag = Port(int64, 'in', protocol='valid')
		self.a_7_real = Port(int64, 'in', protocol='valid')
		self.a_7_imag = Port(int64, 'in', protocol='valid')
		self.c_0_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_0_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_1_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_1_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_2_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_2_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_3_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_3_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_4_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_4_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_5_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_5_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_6_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_6_imag = Port(int64, 'out', protocol='valid', init=0)
		self.c_7_real = Port(int64, 'out', protocol='valid', init=0)
		self.c_7_imag = Port(int64, 'out', protocol='valid', init=0)
                 
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
