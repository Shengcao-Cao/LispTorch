from lt_types import *
from lt_parser import *
import math
import operator as op

def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv,
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq,
        'abs':     abs,
        'append':  op.add,
        'apply':   lambda x, y: x(*y),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:],
        'cons':    lambda x, y: [x] + y,
        'eq?':     op.is_,
        'equal?':  op.eq,
        'length':  len,
        'list':    lambda *x: list(x),
        'list?':   lambda x: isinstance(x,list),
        'map':     lambda x, y: list(map(x, y)),
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [],
        'number?': lambda x: isinstance(x, Number),
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'print':   lambda x: print(lispstr(x)),
        'exit':    lambda: exit(0),
    })
    return env

class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        arg_dict = dict()
        for parm in parms:
            if ':' in parm:
                p, a = parm.split(':')
                arg_dict[p] = atom(a)
        for arg, parm in zip(args, parms):
            if type(arg) is tuple:
                p, a = arg
                arg_dict[p] = a
            else:
                if ':' in parm:
                    p, a = parm.split(':')
                    arg_dict[p] = arg
                else:
                    arg_dict[parm] = arg

        for parm in parms:
            if ':' in parm:
                p, a = parm.split(':')
            else:
                p = parm
            if p not in arg_dict:
                raise Exception('Parameter %s not given.' % p)

        self.update(arg_dict)
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)