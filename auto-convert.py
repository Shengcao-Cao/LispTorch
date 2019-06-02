from lt_parser import *

# torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
# torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)


my_list = [
	'torch.is_tensor(obj)',
	'torch.numel(input)',
	'torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)',
	'torch.as_tensor(data, dtype=None, device=None)',
	'torch.from_numpy(ndarray)',
	'torch.zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',	
	'torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.empty(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)',
	'torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)',
	'torch.cat(tensors, dim=0, out=None)',
]


# (define t_xxx
#	(lambda (x y:blabla z:blabla)
#		(_torch_xxx x y z)))

def convert(string, fout):
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
		fout.write("    '%s': lambda %s: %s(%s),\n" % (name, ', '.join(args).replace('*', ''), tokens[0], args[0] + ', ' + ', '.join(double_args)))
	else:
		args = []
		tokens = [token.replace(',', '') for token in tokens]
		for token in tokens[2:-1]:
			if '=' in token:
				args.append(token.split('=')[0])
			else:
				args.append(token)
		double_args = [arg+'='+arg for arg in args]
		fout.write("    '%s': lambda %s: %s(%s),\n" % (name, ', '.join(args), tokens[0], ', '.join(double_args)))

def lisp_func(string, fout):
	raw_tokens = tokenize(string)
	name = raw_tokens[0].replace('torch.', '_torch_')
	lisp_name = raw_tokens[0].replace('torch.', 't_')


	tokens = []

	for token in raw_tokens:
		token = token.replace('torch.', 't_')
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

	fout.write(func + '\n')


if __name__ == '__main__':
	with open('lt_torch.py', 'w') as fout:
		fout.write('import torch\n')
		fout.write('torch_env = {' + '\n')
		fout.write("    't_size': lambda tensor: tensor.size()," + '\n')
		fout.write("    't_strided': torch.strided," + '\n')
		for func in my_list:
			convert(func, fout)
		fout.write('}')


	with open('torch.lisp', 'w') as fout:
		for func in my_list:
			lisp_func(func, fout)