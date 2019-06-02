import torch
torch_env = {
    't_size': lambda tensor: tensor.size(),
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
}