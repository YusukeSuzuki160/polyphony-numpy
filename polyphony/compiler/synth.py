from collections import defaultdict


class DefaultSynthParamSetter(object):
    def __init__(self):
        pass

## resource condition example
# 1. resource = 'free'
#    - resource is not specified
# 2. resource = {'Add': 1, 'Sub': 1}
#    - resource is specified
#   - resource is fundamental operation
# 3. resource = {'Add': 1, 'Sub': 1, 'Mul': 1, 'listc3r3_add': 1}
#    - resource is specified
#   - resource contains fundamental operation and function

    testbench_params = {
        'scheduling':'parallel',
        'cycle':'minimum',
        "resource":"free",
        "solver":"cp",
        'ii':1,
    }
    scope_params = {
        'scheduling':'parallel',
        'cycle':'minimum',
        # "resource": {"Add": 2},
        "resource":"free",
        "solver":"cp",
        'ii':-1,
        'is_default': True
    }

    def process(self, scope):
        for k, v in scope.synth_params.items():
            if not v:
                if scope.is_testbench():
                    scope.synth_params[k] = self.testbench_params[k]
                else:
                    scope.synth_params[k] = self.scope_params[k]

        for b in scope.traverse_blocks():
            for k, v in b.synth_params.items():
                if not v:
                    b.synth_params[k] = scope.synth_params[k]


def make_synth_params():
    di = defaultdict(str)
    di['scheduling'] = ''
    di['cycle'] = ''
    di['resource'] = ''
    di['ii'] = 0
    di['solver'] = ''
    di['is_default'] = True
    return di


def merge_synth_params(dst_params, src_params):
    for k, v in dst_params.items():
        if not v:
            dst_params[k] = src_params[k]
