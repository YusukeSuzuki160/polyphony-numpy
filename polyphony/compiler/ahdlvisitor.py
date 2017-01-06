class AHDLVisitor:
    def __init__(self):
        pass

    def visit_AHDL_CONST(self, ahdl):
        pass

    def visit_AHDL_VAR(self, ahdl):
        pass

    def visit_AHDL_MEMVAR(self, ahdl):
        pass

    def visit_AHDL_SUBSCRIPT(self, ahdl):
        self.visit(ahdl.memvar)
        self.visit(ahdl.offset)

    def visit_AHDL_OP(self, ahdl):
        self.visit(ahdl.left)
        if ahdl.right:
            self.visit(ahdl.right)

    def visit_AHDL_SYMBOL(self, ahdl):
        pass

    def visit_AHDL_CONCAT(self, ahdl):
        for var in ahdl.varlist:
            self.visit(var)

    def visit_AHDL_NOP(self, ahdl):
        pass

    def visit_AHDL_MOVE(self, ahdl):
        self.visit(ahdl.src)
        self.visit(ahdl.dst)

    def visit_AHDL_STORE(self, ahdl):
        self.visit(ahdl.src)
        self.visit(ahdl.mem)
        self.visit(ahdl.offset)

    def visit_AHDL_LOAD(self, ahdl):
        self.visit(ahdl.mem)
        self.visit(ahdl.dst)
        self.visit(ahdl.offset)

    def visit_AHDL_FIELD_MOVE(self, ahdl):
        self.visit_AHDL_MOVE(ahdl)

    def visit_AHDL_FIELD_STORE(self, ahdl):
        self.visit_AHDL_STORE(ahdl)

    def visit_AHDL_FIELD_LOAD(self, ahdl):
        self.visit_AHDL_LOAD(ahdl)

    def visit_AHDL_POST_PROCESS(self, ahdl):
        method = 'visit_POST_' + ahdl.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(ahdl)

    def visit_AHDL_IF(self, ahdl):
        for cond in ahdl.conds:
            if cond:
                self.visit(cond)
        for codes in ahdl.codes_list:
            for code in codes:
                self.visit(code)

    def visit_AHDL_MODULECALL(self, ahdl):
        for arg in ahdl.args:
            self.visit(arg)

    def visit_AHDL_FUNCALL(self, ahdl):
        for arg in ahdl.args:
            self.visit(arg)

    def visit_AHDL_PROCCALL(self, ahdl):
        for arg in ahdl.args:
            self.visit(arg)

    def visit_AHDL_META(self, ahdl):
        method = 'visit_' + ahdl.metaid
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(ahdl)

    def visit_WAIT_INPUT_READY(self, ahdl):
        if ahdl.codes:
            for code in ahdl.codes:
                self.visit(code)
        self.visit(ahdl.transition)

    def visit_WAIT_OUTPUT_ACCEPT(self, ahdl):
        self.visit(ahdl.transition)

    def visit_WAIT_RET_AND_GATE(self, ahdl):
        self.visit(ahdl.transition)

    def visit_AHDL_META_WAIT(self, ahdl):
        method = 'visit_' + ahdl.metaid
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(ahdl)

    def visit_AHDL_TRANSITION(self, ahdl):
        pass

    def visit_AHDL_TRANSITION_IF(self, ahdl):
        self.visit_AHDL_IF(ahdl)

    def visit(self, ahdl):
        method = 'visit_' + ahdl.__class__.__name__
        visitor = getattr(self, method, None)
        return visitor(ahdl)
