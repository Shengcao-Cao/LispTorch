Number = (int, float)   # A Lisp Number is implemented as a Python int or float
String = str            # A Lisp String is implemented as a Python str
Bool   = bool           # A Lisp Bool is implemented as a Python bool
List   = list           # A Lisp List is implemented as a Python list

class Symbol(str):      # A Lisp Symbol is implemented as a Python str subclass
    def __init__(self, name):
        self = name
# Symbol = str          # A Lisp Symbol is implemented as a Python str