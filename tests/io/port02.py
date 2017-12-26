from polyphony import testbench, module, is_worker_running
#from polyphony.io import Bit
from polyphony.io import Port
from polyphony.typing import bit
from polyphony.timing import clksleep, clkfence, wait_rising, wait_falling


def msg(m, v1, v2):
    print(m, v1, v2)


def other_main(clk1, clk2, out):
    print('other_main')
    while is_worker_running():
        wait_falling(clk1)
        wait_falling(clk2)
        msg('falling', clk1(), clk2())
        out.wr(1)


@module
class Port02:
    def __init__(self):
        self.clk1 = Port(bit, 'in', init=1)
        self.clk2 = Port(bit, 'in', init=1)
        self.out1 = Port(bit, 'out', init=0)
        self.out2 = Port(bit, 'out', init=0)
        self.append_worker(self.main)
        self.append_worker(other_main, self.clk1, self.clk2, self.out2)

    def main(self):
        #print('main')
        while is_worker_running():
            wait_rising(self.clk1)
            wait_rising(self.clk2)
            msg('rising', self.clk1(), self.clk2())
            self.out1.wr(1)

@testbench
def test(p02):
    for i in range(8):
        p02.clk1.wr(1)
        p02.clk2.wr(1)
        p02.clk1.wr(0)
        p02.clk2.wr(0)
        x1 = p02.out1.rd()
        x2 = p02.out2.rd()
        x3 = p02.out1.rd()
        x4 = p02.out2.rd()
    clksleep(2)


p02 = Port02()
test(p02)
