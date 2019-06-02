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

def convert(string):
	tokens = tokenize(string)
	name = tokens[0].replace('torch.', '_torch_')

	if '*' in string:
		args = []
		tokens = [token.replace(',', '') for token in tokens]
		for token in tokens[2:-1]:
			if '=' in token:
				args.append(token.split('=')[0])
			else:
				args.append(token)
		double_args = [arg+'='+arg for arg in args[1:]]
		print("    '%s': lambda %s: %s(%s)," % (name, ', '.join(args).replace('*', ''), tokens[0], args[0] + ', ' + ', '.join(double_args)))
	else:
		args = []
		tokens = [token.replace(',', '') for token in tokens]
		for token in tokens[2:-1]:
			if '=' in token:
				args.append(token.split('=')[0])
			else:
				args.append(token)
		double_args = [arg+'='+arg for arg in args]
		print("    '%s': lambda %s: %s(%s)," % (name, ', '.join(args), tokens[0], ', '.join(double_args)))

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

	func += ' ' + ' '.join(args) + ')))'

	print(func)


if __name__ == '__main__':
	print('{')
	print("    't-strided': torch.strided,")
	for func in my_list:
		convert(func)
	print('}')

	for func in my_list:
		lisp_func(func)