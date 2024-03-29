(define t_is_tensor
	(lambda (obj)
		(_torch_is_tensor obj)))

(define t_numel
	(lambda (input)
		(_torch_numel input)))

(define t_tensor
	(lambda (data device:#n requires_grad:#f pin_memory:#f)
		(_torch_tensor data device requires_grad pin_memory)))

(define t_as_tensor
	(lambda (data device:#n)
		(_torch_as_tensor data device)))

(define t_from_numpy
	(lambda (ndarray)
		(_torch_from_numpy ndarray)))

(define t_zeros
	(lambda (*sizes out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_zeros *sizes out layout device requires_grad)))

(define t_ones
	(lambda (*sizes out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_ones *sizes out layout device requires_grad)))

(define t_arange
	(lambda (start:0 end step:1 out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_arange start end step out layout device requires_grad)))

(define t_range
	(lambda (start:0 end step:1 out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_range start end step out layout device requires_grad)))

(define t_eye
	(lambda (n m:#n out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_eye n m out layout device requires_grad)))

(define t_empty
	(lambda (*sizes out:#n layout:t_strided device:#n requires_grad:#f pin_memory:#f)
		(_torch_empty *sizes out layout device requires_grad pin_memory)))

(define t_full
	(lambda (size fill_value out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_full size fill_value out layout device requires_grad)))

(define t_cat
	(lambda (tensors dim:0 out:#n)
		(_torch_cat tensors dim out)))

(define t_chunk
	(lambda (tensor chunks dim:0)
		(_torch_chunk tensor chunks dim)))

(define t_gather
	(lambda (input dim index out:#n sparse_grad:#f)
		(_torch_gather input dim index out sparse_grad)))

(define t_index_select
	(lambda (input dim index out:#n)
		(_torch_index_select input dim index out)))

(define t_masked_select
	(lambda (input mask out:#n)
		(_torch_masked_select input mask out)))

(define t_narrow
	(lambda (input dimension start length)
		(_torch_narrow input dimension start length)))

(define t_nonzero
	(lambda (input out:#n)
		(_torch_nonzero input out)))

(define t_reshape
	(lambda (input shape)
		(_torch_reshape input shape)))

(define t_split
	(lambda (tensor split_size_or_sections dim:0)
		(_torch_split tensor split_size_or_sections dim)))

(define t_squeeze
	(lambda (input dim:#n out:#n)
		(_torch_squeeze input dim out)))

(define t_stack
	(lambda (seq dim:0 out:#n)
		(_torch_stack seq dim out)))

(define t_t
	(lambda (input)
		(_torch_t input)))

(define t_take
	(lambda (input indices)
		(_torch_take input indices)))

(define t_transpose
	(lambda (input dim0 dim1)
		(_torch_transpose input dim0 dim1)))

(define t_unbind
	(lambda (tensor dim:0)
		(_torch_unbind tensor dim)))

(define t_unsqueeze
	(lambda (input dim out:#n)
		(_torch_unsqueeze input dim out)))

(define t_where
	(lambda (condition x y)
		(_torch_where condition x y)))

(define t_manual_seed
	(lambda (seed)
		(_torch_manual_seed seed)))

(define t_initial_seed
	(lambda ()
		(_torch_initial_seed )))

(define t_normal
	(lambda (mean std:1_0 out:#n)
		(_torch_normal mean std out)))

(define t_rand
	(lambda (*sizes out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_rand *sizes out layout device requires_grad)))

(define t_randint
	(lambda (low:0 high size out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_randint low high size out layout device requires_grad)))

(define t_randn
	(lambda (*sizes out:#n layout:t_strided device:#n requires_grad:#f)
		(_torch_randn *sizes out layout device requires_grad)))

(define t_get_num_threads
	(lambda ()
		(_torch_get_num_threads )))

(define t_set_num_threads
	(lambda (int)
		(_torch_set_num_threads int)))

(define t_abs
	(lambda (input out:#n)
		(_torch_abs input out)))

(define t_acos
	(lambda (input out:#n)
		(_torch_acos input out)))

(define t_add
	(lambda (input value out:#n)
		(_torch_add input value out)))

(define t_asin
	(lambda (input out:#n)
		(_torch_asin input out)))

(define t_atan
	(lambda (input out:#n)
		(_torch_atan input out)))

(define t_atan2
	(lambda (input1 input2 out:#n)
		(_torch_atan2 input1 input2 out)))

(define t_ceil
	(lambda (input out:#n)
		(_torch_ceil input out)))

(define t_clamp
	(lambda (input min max out:#n)
		(_torch_clamp input min max out)))

(define t_cos
	(lambda (input out:#n)
		(_torch_cos input out)))

(define t_cosh
	(lambda (input out:#n)
		(_torch_cosh input out)))

(define t_div
	(lambda (input value out:#n)
		(_torch_div input value out)))

(define t_digamma
	(lambda (input out:#n)
		(_torch_digamma input out)))

(define t_erf
	(lambda (tensor out:#n)
		(_torch_erf tensor out)))

(define t_erfc
	(lambda (input out:#n)
		(_torch_erfc input out)))

(define t_erfinv
	(lambda (input out:#n)
		(_torch_erfinv input out)))

(define t_exp
	(lambda (input out:#n)
		(_torch_exp input out)))

(define t_expm1
	(lambda (input out:#n)
		(_torch_expm1 input out)))

(define t_floor
	(lambda (input out:#n)
		(_torch_floor input out)))

(define t_fmod
	(lambda (input divisor out:#n)
		(_torch_fmod input divisor out)))

(define t_frac
	(lambda (input out:#n)
		(_torch_frac input out)))

(define t_log
	(lambda (input out:#n)
		(_torch_log input out)))

(define t_log10
	(lambda (input out:#n)
		(_torch_log10 input out)))

(define t_log1p
	(lambda (input out:#n)
		(_torch_log1p input out)))

(define t_log2
	(lambda (input out:#n)
		(_torch_log2 input out)))

(define t_mul
	(lambda (input value out:#n)
		(_torch_mul input value out)))

(define t_neg
	(lambda (input out:#n)
		(_torch_neg input out)))

(define t_pow
	(lambda (input exponent out:#n)
		(_torch_pow input exponent out)))

(define t_reciprocal
	(lambda (input out:#n)
		(_torch_reciprocal input out)))

(define t_remainder
	(lambda (input divisor out:#n)
		(_torch_remainder input divisor out)))

(define t_round
	(lambda (input out:#n)
		(_torch_round input out)))

(define t_rsqrt
	(lambda (input out:#n)
		(_torch_rsqrt input out)))

(define t_sigmoid
	(lambda (input out:#n)
		(_torch_sigmoid input out)))

(define t_sign
	(lambda (input out:#n)
		(_torch_sign input out)))

(define t_sin
	(lambda (input out:#n)
		(_torch_sin input out)))

(define t_sinh
	(lambda (input out:#n)
		(_torch_sinh input out)))

(define t_sqrt
	(lambda (input out:#n)
		(_torch_sqrt input out)))

(define t_tan
	(lambda (input out:#n)
		(_torch_tan input out)))

(define t_tanh
	(lambda (input out:#n)
		(_torch_tanh input out)))

(define t_trunc
	(lambda (input out:#n)
		(_torch_trunc input out)))

(define t_argmax
	(lambda (input dim)
		(_torch_argmax input dim)))

(define t_argmin
	(lambda (input dim out:#n)
		(_torch_argmin input dim out)))

(define t_cumprod
	(lambda (input dim out:#n)
		(_torch_cumprod input dim out)))

(define t_cumsum
	(lambda (input dim out:#n)
		(_torch_cumsum input dim out)))

(define t_dist
	(lambda (input other p:2)
		(_torch_dist input other p)))

(define t_logsumexp
	(lambda (input dim out:#n)
		(_torch_logsumexp input dim out)))

(define t_mean
	(lambda (input dim out:#n)
		(_torch_mean input dim out)))

(define t_median
	(lambda (input dim:-1 values:#n indices:#n)
		(_torch_median input dim values indices)))

(define t_mode
	(lambda (input dim:-1 values:#n indices:#n)
		(_torch_mode input dim values indices)))

(define t_norm
	(lambda (input p:"fro" dim:#n out:#n)
		(_torch_norm input p dim out)))

(define t_prod
	(lambda (input dim)
		(_torch_prod input dim)))

(define t_std
	(lambda (input dim unbiased:#t out:#n)
		(_torch_std input dim unbiased out)))

(define t_sum
	(lambda (input dim)
		(_torch_sum input dim)))

(define t_unique
	(lambda (input sorted:#t return_inverse:#f return_counts:#f dim:#n)
		(_torch_unique input sorted return_inverse return_counts dim)))

(define t_unique_consecutive
	(lambda (input return_inverse:#f return_counts:#f dim:#n)
		(_torch_unique_consecutive input return_inverse return_counts dim)))

(define t_var
	(lambda (input dim unbiased:#t out:#n)
		(_torch_var input dim unbiased out)))

(define t_argsort
	(lambda (input dim:-1 descending:#f out:#n)
		(_torch_argsort input dim descending out)))

(define t_allclose
	(lambda (self other rtol:1e-05 atol:1e-08 equal_nan:#f)
		(_torch_allclose self other rtol atol equal_nan)))

(define t_eq
	(lambda (input other out:#n)
		(_torch_eq input other out)))

(define t_equal
	(lambda (tensor1 tensor2)
		(_torch_equal tensor1 tensor2)))

(define t_ge
	(lambda (input other out:#n)
		(_torch_ge input other out)))

(define t_gt
	(lambda (input other out:#n)
		(_torch_gt input other out)))

(define t_isfinite
	(lambda (tensor)
		(_torch_isfinite tensor)))

(define t_isinf
	(lambda (tensor)
		(_torch_isinf tensor)))

(define t_isnan
	(lambda (tensor)
		(_torch_isnan tensor)))

(define t_le
	(lambda (input other out:#n)
		(_torch_le input other out)))

(define t_lt
	(lambda (input other out:#n)
		(_torch_lt input other out)))

(define t_max
	(lambda (input dim out:#n)
		(_torch_max input dim out)))

(define t_min
	(lambda (input dim out:#n)
		(_torch_min input dim out)))

(define t_ne
	(lambda (input other out:#n)
		(_torch_ne input other out)))

(define t_sort
	(lambda (input dim:-1 descending:#f out:#n)
		(_torch_sort input dim descending out)))

(define t_topk
	(lambda (input k dim:#n largest:#t sorted:#t out:#n)
		(_torch_topk input k dim largest sorted out)))

(define t_fft
	(lambda (input signal_ndim normalized:#f)
		(_torch_fft input signal_ndim normalized)))

(define t_ifft
	(lambda (input signal_ndim normalized:#f)
		(_torch_ifft input signal_ndim normalized)))

(define t_rfft
	(lambda (input signal_ndim normalized:#f onesided:#t)
		(_torch_rfft input signal_ndim normalized onesided)))

(define t_irfft
	(lambda (input signal_ndim normalized:#f onesided:#t signal_sizes:#n)
		(_torch_irfft input signal_ndim normalized onesided signal_sizes)))

(define t_stft
	(lambda (input n_fft hop_length:#n win_length:#n window:#n center:#t pad_mode:"reflect" normalized:#f onesided:#t)
		(_torch_stft input n_fft hop_length win_length window center pad_mode normalized onesided)))

(define t_bartlett_window
	(lambda (window_length periodic:#t layout:t_strided device:#n requires_grad:#f)
		(_torch_bartlett_window window_length periodic layout device requires_grad)))

(define t_blackman_window
	(lambda (window_length periodic:#t layout:t_strided device:#n requires_grad:#f)
		(_torch_blackman_window window_length periodic layout device requires_grad)))

(define t_hamming_window
	(lambda (window_length periodic:#t alpha:0_54 beta:0_46 layout:t_strided device:#n requires_grad:#f)
		(_torch_hamming_window window_length periodic alpha beta layout device requires_grad)))

(define t_cross
	(lambda (input other dim:-1 out:#n)
		(_torch_cross input other dim out)))

(define t_diag
	(lambda (input diagonal:0 out:#n)
		(_torch_diag input diagonal out)))

(define t_flatten
	(lambda (input start_dim:0 end_dim:-1)
		(_torch_flatten input start_dim end_dim)))

(define t_flip
	(lambda (input dims)
		(_torch_flip input dims)))

(define t_tensordot
	(lambda (a b dims:2)
		(_torch_tensordot a b dims)))

(define t_trace
	(lambda (input)
		(_torch_trace input)))

(define t_addbmm
	(lambda (beta:1 mat alpha:1 batch1 batch2 out:#n)
		(_torch_addbmm beta mat alpha batch1 batch2 out)))

(define t_addmm
	(lambda (beta:1 mat alpha:1 mat1 mat2 out:#n)
		(_torch_addmm beta mat alpha mat1 mat2 out)))

(define t_bmm
	(lambda (batch1 batch2 out:#n)
		(_torch_bmm batch1 batch2 out)))

(define t_dot
	(lambda (tensor1 tensor2)
		(_torch_dot tensor1 tensor2)))

(define t_eig
	(lambda (a eigenvectors:#f out:#n)
		(_torch_eig a eigenvectors out)))

(define t_geqrf
	(lambda (input out:#n)
		(_torch_geqrf input out)))

(define t_inverse
	(lambda (input out:#n)
		(_torch_inverse input out)))

(define t_det
	(lambda (A)
		(_torch_det A)))

(define t_logdet
	(lambda (A)
		(_torch_logdet A)))

(define t_matmul
	(lambda (tensor1 tensor2 out:#n)
		(_torch_matmul tensor1 tensor2 out)))

(define t_mm
	(lambda (mat1 mat2 out:#n)
		(_torch_mm mat1 mat2 out)))

(define t_mv
	(lambda (mat vec out:#n)
		(_torch_mv mat vec out)))

(define t_utils_data_DataLoader
	(lambda (dataset batch_size:1 shuffle:#f sampler:#n batch_sampler:#n num_workers:0 pin_memory:#f drop_last:#f timeout:0 worker_init_fn:#n)
		(_torch_utils_data_DataLoader dataset batch_size shuffle sampler batch_sampler num_workers pin_memory drop_last timeout worker_init_fn)))

(define t_optim_SGD
	(lambda (params lr momentum:0 dampening:0 weight_decay:0 nesterov:#f)
		(_torch_optim_SGD params lr momentum dampening weight_decay nesterov)))

(define t_optim_Adam
	(lambda (params lr:0_001 eps:1e-08 weight_decay:0 amsgrad:#f)
		(_torch_optim_Adam params lr eps weight_decay amsgrad)))

(define t_optim_Adagrad
	(lambda (params lr:0_01 lr_decay:0 weight_decay:0 initial_accumulator_value:0)
		(_torch_optim_Adagrad params lr lr_decay weight_decay initial_accumulator_value)))

(define t_nn_Sequential
	(lambda (*args)
		(_torch_nn_Sequential *args)))

(define t_nn_Conv2d
	(lambda (in_channels out_channels kernel_size stride:1 padding:0 dilation:1 groups:1 bias:#t)
		(_torch_nn_Conv2d in_channels out_channels kernel_size stride padding dilation groups bias)))

(define t_nn_ReLU
	(lambda (inplace:#f)
		(_torch_nn_ReLU inplace)))

(define t_nn_MaxPool2d
	(lambda (kernel_size stride:#n padding:0 dilation:1 return_indices:#f ceil_mode:#f)
		(_torch_nn_MaxPool2d kernel_size stride padding dilation return_indices ceil_mode)))

(define tv_transforms_Compose
	(lambda (transforms)
		(_torchvision_transforms_Compose transforms)))

(define tv_transforms_ToTensor
	(lambda ()
		(_torchvision_transforms_ToTensor )))

(define tv_transforms_Normalize
	(lambda (mean std inplace:#f)
		(_torchvision_transforms_Normalize mean std inplace)))

(define t_nn_Linear
	(lambda (in_features out_features bias:#t)
		(_torch_nn_Linear in_features out_features bias)))

(define t_nn_LogSoftmax
	(lambda (dim:#n)
		(_torch_nn_LogSoftmax dim)))

(define t_nn_functional_nll_loss
	(lambda (input target weight:#n size_average:#n ignore_index:-100 reduce:#n)
		(_torch_nn_functional_nll_loss input target weight size_average ignore_index reduce)))

