from lt_types import *
from lt_parser import *
from lt_env import *

import argparse
import traceback
import sys

def finished(tokens):
    count = 0
    for token in tokens:
        if token == '(':
            count += 1
        elif token == ')':
            count -=1
        if count < 0:
            return True
    if count == 0:
        return True
    else:
        return False

def repl(env, prompt='>>> '):
    "A prompt-read-eval-print loop."
    all_tokens = []
    while True:
        try:
            tokens = tokenize(input(prompt))
            all_tokens += tokens
            while finished(all_tokens) and len(all_tokens) > 0:
                parse_result, all_tokens = parse(all_tokens)
                val = eval(parse_result, env)
                if val is not None:
                    print(lispstr(val))
        except SystemExit:
            raise
        except EOFError:
            raise
        except KeyboardInterrupt:
            raise
        except:
            # print('Traceback (most recent call last):')
            # info = sys.exc_info()
            # traceback.print_tb(info[2])
            # print('{}: {}'.format(info[0].__name__, info[1]))
            traceback.print_exc()
            all_tokens = []

def repl_file(file, env):
    "A file-based repl."
    all_tokens = []
    with open(file, 'r') as f:
        for line in f.readlines():
            tokens = tokenize(line)
            all_tokens += tokens
            while finished(all_tokens) and len(all_tokens) > 0:
                parse_result, all_tokens = parse(all_tokens)
                _ = eval(parse_result, env)

class Procedure(object):
    "A user-defined Lisp procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args):
        return eval(self.body, Env(self.parms, args, self.env))

def eval(x, env):
    "Evaluate an expression in an environment."
    while True:
        if isinstance(x, Symbol):      # variable reference
            return env.find(x)[x]
        elif not isinstance(x, List):  # constant literal
            return x
        elif x[0] == 'quote':          # (quote exp)
            (_, exp) = x
            return exp
        elif x[0] == 'if':             # (if test conseq alt)
            (_, test, conseq, alt) = x
            x = (conseq if eval(test, env) else alt)
        elif x[0] == 'define':         # (define var exp)
            (_, var, exp) = x
            env[var] = eval(exp, env)
            return None
        elif x[0] == 'let':            # (let var exp)
            (_, vars, exp) = x
            parms = [var[0] for var in vars]
            args = [var[1] for var in vars]
            env = Env(parms, args, env)
            return eval(exp, env)
        elif x[0] == 'set!':           # (set! var exp)
            (_, var, exp) = x
            env.find(var)[var] = eval(exp, env)
            return None
        elif x[0] == 'lambda':         # (lambda (var...) body)
            (_, parms, body) = x
            return Procedure(parms, body, env)
        elif x[0] == 'include':        # (include header)
            header = x[1] + '.lisp'
            repl_file(header, env)
            return None
        else:                          # (proc arg...)
            try:
                proc = eval(x[0], env)
                args = []
            except:
                print('x[0]:', x)
                raise

            pointer = 1

            while pointer < len(x):
                exp = x[pointer]
                if type(exp) is str and ':' in exp:
                    p, a = exp.split(':')
                    if a is '':
                        pointer += 1
                        exp = x[pointer]
                        try:
                            args.append((p, eval(exp, env)))
                        except:
                            print('exp:', exp)
                            raise
                    else:
                        try:
                            args.append((p, eval(atom(a), env)))
                        except:
                            print('exp:', exp)
                            raise
                else:
                    try:
                        args.append(eval(exp, env))
                    except:
                        print('exp:', exp)
                        raise
                pointer += 1

            if isinstance(proc, Procedure):
                x = proc.body
                env = Env(proc.parms, args, proc.env)
            else:
                try:
                    return proc(*args)
                except:
                    print('proc:', proc)
                    print('args:', args)
                    raise

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='LispTorch: Let\'s use PyTorch in the classic Lisp style!')
    arg_parser.add_argument('-f', '--file', type=str, default='')
    args = arg_parser.parse_args()
    global_env = standard_env()
    if args.file == '':
        repl(global_env)
    else:
        repl_file(args.file, global_env)