import torch
torch_env = {
    't-strided': torch.strided,
    '_torch_is_tensor': lambda obj: torch.is_tensor(obj=obj),
    '_torch_numel': lambda input: torch.numel(input=input),
    '_torch_tensor': lambda data, dtype, device, requires_grad, pin_memory: torch.tensor(data=data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory),
    '_torch_as_tensor': lambda data, dtype, device: torch.as_tensor(data=data, dtype=dtype, device=device),
    '_torch_from_numpy': lambda ndarray: torch.from_numpy(ndarray=ndarray),
    '_torch_zeros': lambda sizes, out, dtype, layout, device, requires_grad: torch.zeros(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_zeros_like': lambda input, dtype, layout, device, requires_grad: torch.zeros_like(input=input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
    '_torch_ones': lambda sizes, out, dtype, layout, device, requires_grad: torch.ones(*sizes, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
}