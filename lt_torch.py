import torch
import torchvision
class MyReshape(torch.nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(self.shape)

torch_env = {
    't_nn_Reshape': MyReshape
	'tv_datasets_MNIST': lambda train=True, transform=None, target_transform=None, download=False: torchvision.datasets.MNIST('../data', train=train, transform=transform, target_transform=target_transform, download=download)    't_iter': lambda _list: iter(_list),
    't_next': lambda iter: next(iter),
    't_zero_grad': lambda optimizer: optimizer.zero_grad(),
    't_step': lambda optimizer: optimizer.step(),
    't_backward': lambda loss: loss.backward(),
    't_view': lambda tensor, _list: tensor.view(*_list),
    't_size': lambda tensor: tensor.size(),
    't_item': lambda tensor: tensor.item(),
    't_strided': torch.strided,
    '_torch_is_tensor': lambda obj: torch.is_tensor(obj=obj),
    '_torch_numel': lambda input: torch.numel(input=input),
    '_torch_tensor': lambda data, dtype, device, requires_grad, pin_memory: torch.tensor(data=data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory),
    '_torch_as_tensor': lambda data, dtype, device: torch.as_tensor(data=data, dtype=dtype, device=device),
    '_torch_from_numpy': lambda ndarray: torch.from_numpy(ndarray=ndarray),
    '_torch_zeros': lambda sizes, out, dtype, layout, device, requires_grad: torch.zeros(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_ones': lambda sizes, out, dtype, layout, device, requires_grad: torch.ones(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_arange': lambda start, end, step, out, dtype, layout, device, requires_grad: torch.arange(start=start, end=end, step=step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_range': lambda start, end, step, out, dtype, layout, device, requires_grad: torch.range(start=start, end=end, step=step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_eye': lambda n, m, out, dtype, layout, device, requires_grad: torch.eye(n=n, m=m, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_empty': lambda sizes, out, dtype, layout, device, requires_grad, pin_memory: torch.empty(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory),
    '_torch_full': lambda size, fill_value, out, dtype, layout, device, requires_grad: torch.full(size=size, fill_value=fill_value, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_cat': lambda tensors, dim, out: torch.cat(tensors=tensors, dim=dim, out=out),
    '_torch_chunk': lambda tensor, chunks, dim: torch.chunk(tensor=tensor, chunks=chunks, dim=dim),
    '_torch_gather': lambda input, dim, index, out, sparse_grad: torch.gather(input=input, dim=dim, index=index, out=out, sparse_grad=sparse_grad),
    '_torch_index_select': lambda input, dim, index, out: torch.index_select(input=input, dim=dim, index=index, out=out),
    '_torch_masked_select': lambda input, mask, out: torch.masked_select(input=input, mask=mask, out=out),
    '_torch_narrow': lambda input, dimension, start, length: torch.narrow(input=input, dimension=dimension, start=start, length=length),
    '_torch_nonzero': lambda input, out: torch.nonzero(input=input, out=out),
    '_torch_reshape': lambda input, shape: torch.reshape(input=input, shape=shape),
    '_torch_split': lambda tensor, split_size_or_sections, dim: torch.split(tensor=tensor, split_size_or_sections=split_size_or_sections, dim=dim),
    '_torch_squeeze': lambda input, dim, out: torch.squeeze(input=input, dim=dim, out=out),
    '_torch_stack': lambda seq, dim, out: torch.stack(seq=seq, dim=dim, out=out),
    '_torch_t': lambda input: torch.t(input=input),
    '_torch_take': lambda input, indices: torch.take(input=input, indices=indices),
    '_torch_transpose': lambda input, dim0, dim1: torch.transpose(input=input, dim0=dim0, dim1=dim1),
    '_torch_unbind': lambda tensor, dim: torch.unbind(tensor=tensor, dim=dim),
    '_torch_unsqueeze': lambda input, dim, out: torch.unsqueeze(input=input, dim=dim, out=out),
    '_torch_where': lambda condition, x, y: torch.where(condition=condition, x=x, y=y),
    '_torch_manual_seed': lambda seed: torch.manual_seed(seed=seed),
    '_torch_initial_seed': lambda : torch.initial_seed(),
    '_torch_normal': lambda mean, std, out: torch.normal(mean=mean, std=std, out=out),
    '_torch_rand': lambda sizes, out, dtype, layout, device, requires_grad: torch.rand(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_randint': lambda low, high, size, out, dtype, layout, device, requires_grad: torch.randint(low=low, high=high, size=size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_randn': lambda sizes, out, dtype, layout, device, requires_grad: torch.randn(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_get_num_threads': lambda : torch.get_num_threads(),
    '_torch_set_num_threads': lambda int: torch.set_num_threads(int=int),
    '_torch_abs': lambda input, out: torch.abs(input=input, out=out),
    '_torch_acos': lambda input, out: torch.acos(input=input, out=out),
    '_torch_add': lambda input, value, out: torch.add(input=input, value=value, out=out),
    '_torch_asin': lambda input, out: torch.asin(input=input, out=out),
    '_torch_atan': lambda input, out: torch.atan(input=input, out=out),
    '_torch_atan2': lambda input1, input2, out: torch.atan2(input1=input1, input2=input2, out=out),
    '_torch_ceil': lambda input, out: torch.ceil(input=input, out=out),
    '_torch_clamp': lambda input, min, max, out: torch.clamp(input=input, min=min, max=max, out=out),
    '_torch_cos': lambda input, out: torch.cos(input=input, out=out),
    '_torch_cosh': lambda input, out: torch.cosh(input=input, out=out),
    '_torch_div': lambda input, value, out: torch.div(input=input, value=value, out=out),
    '_torch_digamma': lambda input, out: torch.digamma(input=input, out=out),
    '_torch_erf': lambda tensor, out: torch.erf(tensor=tensor, out=out),
    '_torch_erfc': lambda input, out: torch.erfc(input=input, out=out),
    '_torch_erfinv': lambda input, out: torch.erfinv(input=input, out=out),
    '_torch_exp': lambda input, out: torch.exp(input=input, out=out),
    '_torch_expm1': lambda input, out: torch.expm1(input=input, out=out),
    '_torch_floor': lambda input, out: torch.floor(input=input, out=out),
    '_torch_fmod': lambda input, divisor, out: torch.fmod(input=input, divisor=divisor, out=out),
    '_torch_frac': lambda input, out: torch.frac(input=input, out=out),
    '_torch_log': lambda input, out: torch.log(input=input, out=out),
    '_torch_log10': lambda input, out: torch.log10(input=input, out=out),
    '_torch_log1p': lambda input, out: torch.log1p(input=input, out=out),
    '_torch_log2': lambda input, out: torch.log2(input=input, out=out),
    '_torch_mul': lambda input, value, out: torch.mul(input=input, value=value, out=out),
    '_torch_neg': lambda input, out: torch.neg(input=input, out=out),
    '_torch_pow': lambda input, exponent, out: torch.pow(input=input, exponent=exponent, out=out),
    '_torch_reciprocal': lambda input, out: torch.reciprocal(input=input, out=out),
    '_torch_remainder': lambda input, divisor, out: torch.remainder(input=input, divisor=divisor, out=out),
    '_torch_round': lambda input, out: torch.round(input=input, out=out),
    '_torch_rsqrt': lambda input, out: torch.rsqrt(input=input, out=out),
    '_torch_sigmoid': lambda input, out: torch.sigmoid(input=input, out=out),
    '_torch_sign': lambda input, out: torch.sign(input=input, out=out),
    '_torch_sin': lambda input, out: torch.sin(input=input, out=out),
    '_torch_sinh': lambda input, out: torch.sinh(input=input, out=out),
    '_torch_sqrt': lambda input, out: torch.sqrt(input=input, out=out),
    '_torch_tan': lambda input, out: torch.tan(input=input, out=out),
    '_torch_tanh': lambda input, out: torch.tanh(input=input, out=out),
    '_torch_trunc': lambda input, out: torch.trunc(input=input, out=out),
    '_torch_argmax': lambda input, dim, keepdim: torch.argmax(input=input, dim=dim, keepdim=keepdim),
    '_torch_argmin': lambda input, dim, keepdim, out: torch.argmin(input=input, dim=dim, keepdim=keepdim, out=out),
    '_torch_cumprod': lambda input, dim, out, dtype: torch.cumprod(input=input, dim=dim, out=out, dtype=dtype),
    '_torch_cumsum': lambda input, dim, out, dtype: torch.cumsum(input=input, dim=dim, out=out, dtype=dtype),
    '_torch_dist': lambda input, other, p: torch.dist(input=input, other=other, p=p),
    '_torch_logsumexp': lambda input, dim, keepdim, out: torch.logsumexp(input=input, dim=dim, keepdim=keepdim, out=out),
    '_torch_mean': lambda input, dim, keepdim, out: torch.mean(input=input, dim=dim, keepdim=keepdim, out=out),
    '_torch_median': lambda input, dim, keepdim, values, indices: torch.median(input=input, dim=dim, keepdim=keepdim, values=values, indices=indices),
    '_torch_mode': lambda input, dim, keepdim, values, indices: torch.mode(input=input, dim=dim, keepdim=keepdim, values=values, indices=indices),
    '_torch_norm': lambda input, p, dim, keepdim, out, dtype: torch.norm(input=input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype),
    '_torch_prod': lambda input, dim, keepdim, dtype: torch.prod(input=input, dim=dim, keepdim=keepdim, dtype=dtype),
    '_torch_std': lambda input, dim, keepdim, unbiased, out: torch.std(input=input, dim=dim, keepdim=keepdim, unbiased=unbiased, out=out),
    '_torch_sum': lambda input, dim, keepdim, dtype: torch.sum(input=input, dim=dim, keepdim=keepdim, dtype=dtype),
    '_torch_unique': lambda input, sorted, return_inverse, return_counts, dim: torch.unique(input=input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim),
    '_torch_unique_consecutive': lambda input, return_inverse, return_counts, dim: torch.unique_consecutive(input=input, return_inverse=return_inverse, return_counts=return_counts, dim=dim),
    '_torch_var': lambda input, dim, keepdim, unbiased, out: torch.var(input=input, dim=dim, keepdim=keepdim, unbiased=unbiased, out=out),
    '_torch_argsort': lambda input, dim, descending, out: torch.argsort(input=input, dim=dim, descending=descending, out=out),
    '_torch_allclose': lambda self, other, rtol, atol, equal_nan: torch.allclose(self=self, other=other, rtol=rtol, atol=atol, equal_nan=equal_nan),
    '_torch_eq': lambda input, other, out: torch.eq(input=input, other=other, out=out),
    '_torch_equal': lambda tensor1, tensor2: torch.equal(tensor1=tensor1, tensor2=tensor2),
    '_torch_ge': lambda input, other, out: torch.ge(input=input, other=other, out=out),
    '_torch_gt': lambda input, other, out: torch.gt(input=input, other=other, out=out),
    '_torch_isfinite': lambda tensor: torch.isfinite(tensor=tensor),
    '_torch_isinf': lambda tensor: torch.isinf(tensor=tensor),
    '_torch_isnan': lambda tensor: torch.isnan(tensor=tensor),
    '_torch_le': lambda input, other, out: torch.le(input=input, other=other, out=out),
    '_torch_lt': lambda input, other, out: torch.lt(input=input, other=other, out=out),
    '_torch_max': lambda input, dim, keepdim, out: torch.max(input=input, dim=dim, keepdim=keepdim, out=out),
    '_torch_min': lambda input, dim, keepdim, out: torch.min(input=input, dim=dim, keepdim=keepdim, out=out),
    '_torch_ne': lambda input, other, out: torch.ne(input=input, other=other, out=out),
    '_torch_sort': lambda input, dim, descending, out: torch.sort(input=input, dim=dim, descending=descending, out=out),
    '_torch_topk': lambda input, k, dim, largest, sorted, out: torch.topk(input=input, k=k, dim=dim, largest=largest, sorted=sorted, out=out),
    '_torch_fft': lambda input, signal_ndim, normalized: torch.fft(input=input, signal_ndim=signal_ndim, normalized=normalized),
    '_torch_ifft': lambda input, signal_ndim, normalized: torch.ifft(input=input, signal_ndim=signal_ndim, normalized=normalized),
    '_torch_rfft': lambda input, signal_ndim, normalized, onesided: torch.rfft(input=input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided),
    '_torch_irfft': lambda input, signal_ndim, normalized, onesided, signal_sizes: torch.irfft(input=input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, signal_sizes=signal_sizes),
    '_torch_stft': lambda input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided: torch.stft(input=input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided),
    '_torch_bartlett_window': lambda window_length, periodic, dtype, layout, device, requires_grad: torch.bartlett_window(window_length=window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_blackman_window': lambda window_length, periodic, dtype, layout, device, requires_grad: torch.blackman_window(window_length=window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_hamming_window': lambda window_length, periodic, alpha, beta, dtype, layout, device, requires_grad: torch.hamming_window(window_length=window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_cross': lambda input, other, dim, out: torch.cross(input=input, other=other, dim=dim, out=out),
    '_torch_diag': lambda input, diagonal, out: torch.diag(input=input, diagonal=diagonal, out=out),
    '_torch_flatten': lambda input, start_dim, end_dim: torch.flatten(input=input, start_dim=start_dim, end_dim=end_dim),
    '_torch_flip': lambda input, dims: torch.flip(input=input, dims=dims),
    '_torch_tensordot': lambda a, b, dims: torch.tensordot(a=a, b=b, dims=dims),
    '_torch_trace': lambda input: torch.trace(input=input),
    '_torch_addbmm': lambda beta, mat, alpha, batch1, batch2, out: torch.addbmm(beta=beta, mat=mat, alpha=alpha, batch1=batch1, batch2=batch2, out=out),
    '_torch_addmm': lambda beta, mat, alpha, mat1, mat2, out: torch.addmm(beta=beta, mat=mat, alpha=alpha, mat1=mat1, mat2=mat2, out=out),
    '_torch_bmm': lambda batch1, batch2, out: torch.bmm(batch1=batch1, batch2=batch2, out=out),
    '_torch_dot': lambda tensor1, tensor2: torch.dot(tensor1=tensor1, tensor2=tensor2),
    '_torch_eig': lambda a, eigenvectors, out: torch.eig(a=a, eigenvectors=eigenvectors, out=out),
    '_torch_geqrf': lambda input, out: torch.geqrf(input=input, out=out),
    '_torch_inverse': lambda input, out: torch.inverse(input=input, out=out),
    '_torch_det': lambda A: torch.det(A=A),
    '_torch_logdet': lambda A: torch.logdet(A=A),
    '_torch_matmul': lambda tensor1, tensor2, out: torch.matmul(tensor1=tensor1, tensor2=tensor2, out=out),
    '_torch_mm': lambda mat1, mat2, out: torch.mm(mat1=mat1, mat2=mat2, out=out),
    '_torch_mv': lambda mat, vec, out: torch.mv(mat=mat, vec=vec, out=out),
    '_torch_utils_data_DataLoader': lambda dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, pin_memory, drop_last, timeout, worker_init_fn: torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn),
    '_torch_optim_SGD': lambda params, lr, momentum, dampening, weight_decay, nesterov: torch.optim.SGD(params=params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov),
    '_torch_optim_Adam': lambda params, lr, eps, weight_decay, amsgrad: torch.optim.Adam(params=params, lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad),
    '_torch_optim_Adagrad': lambda params, lr, lr_decay, weight_decay, initial_accumulator_value: torch.optim.Adagrad(params=params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value),
    '_torch_nn_Sequential': lambda args: torch.nn.Sequential(*args, ),
    '_torch_nn_Conv2d': lambda in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias: torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
    '_torch_nn_ReLU': lambda inplace: torch.nn.ReLU(inplace=inplace),
    '_torch_nn_MaxPool2d': lambda kernel_size, stride, padding, dilation, return_indices, ceil_mode: torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode),
    '_torchvision_transforms_Compose': lambda transforms: torchvision.transforms.Compose(transforms=transforms),
    '_torchvision_transforms_ToTensor': lambda : torchvision.transforms.ToTensor(),
    '_torchvision_transforms_Normalize': lambda mean, std, inplace: torchvision.transforms.Normalize(mean=mean, std=std, inplace=inplace),
}