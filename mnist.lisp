(include torch)

(define train
    (lambda (model device train_itr optimizer iteration)
            (begin (if (<= iteration 0)
                       model
                       (begin (define batch (t_next train_itr))
                              (define data (t_to (car batch) device))
                              (define target (t_to (car (cdr batch)) device))
                              (t_zero_grad optimizer)
                              (define output (model data))
                              (define loss (t_F_nll_loss output target))
                              (t_backward loss)
                              (t_step optimizer)
                              (train model device train_itr optimizer (- iteration 1)))))))

(define test
    (lambda (model device test_itr iteration correct total)
            (begin (if (<= iteration 0)
                       (print (/ (* 100.00 correct) total))
                       (begin (define batch (t_next test_itr))
                              (define data (t_to (car batch) device))
                              (define target (t_to (car (cdr batch)) device))
                              (define output (model data))
                              (define pred (t_argmax output dim:1 keepdim:#t))
                              (define correct_i (t_item (t_sum (t_eq pred (t_view target (t_size pred))))))
                              (define total_i (t_size target 0))
                              (test model device test_itr (- iteration 1) (+ correct correct_i) (+ total total_i)))))))

(define train_test
    (lambda (model device train_itr test_itr optimizer round)
            (begin (if (<= round 0))
                   model
                   (begin (train model train_itr optimizer 50)
                       (if (= (% round 10) 0)
                           (test model device test_itr 50)
                           #n)
                       (train_test model device train_itr test_itr optimizer (- round 1))
                       (train model)))))

(define transform (tv_transforms_Compose (list ((tv_transforms_ToTensor) (tv_transforms_Normalize (list 0.1307) (list 0.3081))))))
(define train_dataset (tv_datasets_MNIST "../data" train:#t transform:transform))
(define train_loader (t_utils_data_DataLoader train_dataset batch_size:128 shuffle:#t num_workers:4 pin_memory:#t))
(define train_itr (iter train_loader))
(define test_dataset (tv_datasets_MNIST "../data" train:#f transform:transform))
(define test_loader (t_utils_data_DataLoader train_dataset batch_size:128 shuffle:#f num_workers:4 pin_memory:#t))
(define test_itr (iter test_loader))

(define model
  (t_nn_Sequential (list 
    (t_nn_Conv2d (list 1 20 5 1))
    (t_nn_ReLU)
    (t_nn_Conv2d (list ))

(define optimizer (t_optim_SGD (t_parameters model) lr:0.01 momentum:0.9))

(train_test model device train_itr optimizer 100)