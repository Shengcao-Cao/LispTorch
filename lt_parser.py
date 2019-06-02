from lt_types import *

def parse(program):
    "Read a Lisp expression from a string."
    return read_from_tokens(program)

def tokenize(s):
    "Convert a string into a list of tokens."
    return s.replace('(',' ( ').replace(')',' ) ').split()

def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        return None, []
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens)[0])
        tokens.pop(0)
        return L, tokens
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token), tokens[1:]

def atom(token):
    "Numbers become numbers; every other token is a symbol."
    if token == '#t':
        return True
    elif token == '#f':
        return False
    elif token == '#n':
        return None
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            return Symbol(token)

def lispstr(exp):
    "Convert a Python object back into a Lisp-readable string."
    if isinstance(exp, List):
        return '(' + ' '.join(map(lispstr, exp)) + ')'
    else:
        if exp is True:
            return '#t'
        elif exp is False:
            return '#f'
        elif exp is None:
            return '#n'
        return str(exp)