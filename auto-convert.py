from lt_parser import *

my_list = [
	'torch.is_tensor(obj)',
	'torch.numel(input)',
	'torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)',
	'torch.as_tensor(data, dtype=None, device=None)',
	'torch.from_numpy(ndarray)',
	'torch.zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)',
	'torch.ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
]


# (define t-xxx
#	(lambda (x y:blabla z:blabla)
#		(_torch_xxx x y z)))

# '_torch_zeros': lambda args, **kwargs: torch.zeros(*args, **kwargs),

def convert(string):
	tokens = tokenize(string)
	name = tokens[0].replace('torch.', '_torch_')

	if '*' in string:
		print("'%s': lambda args, **kwargs: %s(*args, **kwargs)," % (name, tokens[0]))
	else:
		print("'%s': %s," % (name, tokens[0]))

def lisp_func(string):
	raw_tokens = tokenize(string)
	name = raw_tokens[0].replace('torch.', '_torch_')
	lisp_name = raw_tokens[0].replace('torch.', 't-')


	tokens = []

	for token in raw_tokens:
		token = token.replace('torch.', 't-')
		token = token.replace('=', ':')
		token = token.replace('False', '#f')
		token = token.replace('True', '#t')
		token = token.replace('None', '#n')
		token = token.replace(',', '')
		tokens.append(token)

	func = '(define %s (lambda (' % lisp_name

	func += ' '.join(tokens[2:-1]) + ') (' + name

	args = []

	for token in tokens[2:-1]:
		if ':' in token:
			args.append(token.split(':')[0])
		else:
			args.append(token)

	func += ' ' + ' '.join(args) + ')'

	print(func)


if __name__ == '__main__':
	print('{')
	for func in my_list:
		convert(func)
	print('}')

	for func in my_list:
		lisp_func(func)