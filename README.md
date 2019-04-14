# LispTorch

*Shengcao Cao & Zhiqing Sun*

Let's use PyTorch in the classic Lisp style!

## Requirements

- Python 3
- PyTorch 1.0

## Usage

```
$ python main.py
>>> (define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
>>> (fact 100)
93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
>>> ...
```

## Reference

- [(How to Write a (Lisp) Interpreter (in Python))](http://norvig.com/lispy.html)