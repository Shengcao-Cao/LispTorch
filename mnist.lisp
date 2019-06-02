(include torch)

(define train
    (lambda (model train_itr optimizer iteration)
            (begin (if (<= iteration 0)
                       model
                       (begin (define batch (t_next train_itr))
                              (define data (car batch))
                              (define target (car (cdr batch)))
                              (t_zero_grad optimizer)
                              (define output (model data))
                              (define loss (t_nn_functional_nll_loss output target))
                              (t_backward loss)
                              (t_step optimizer)
                              (train model train_itr optimizer (- iteration 1)))))))

(define test
    (lambda (model test_itr iteration correct total)
            (begin (if (<= iteration 0)
                       (print (/ (* 100.00 correct) total))
                       (begin (define batch (t_next test_itr))
                              (define data (car batch))
                              (define target (car (cdr batch)))
                              (define output (model data))
                              (define pred (t_argmax output dim:1 keepdim:#t))
                              (define correct_i (t_item (t_sum (t_eq pred (t_view target (t_size pred))) dim:0)))
                              (define total_i (t_size target 0))
                              (test model test_itr (- iteration 1) (+ correct correct_i) (+ total total_i)))))))

(define train_test
    (lambda (model train_itr test_itr optimizer round)
            (begin (if (<= round 0)
                       model
                       (begin (train model train_itr optimizer 50)
                           (if (= (% round 10) 0)
                               (test model test_itr 50 0 0)
                               #n)
                           (train_test model train_itr test_itr optimizer (- round 1))
                           (train model))))))

(define transform (tv_transforms_Compose (list (tv_transforms_ToTensor) (tv_transforms_Normalize (list 0.1307) (list 0.3081)))))
(define train_dataset (tv_datasets_MNIST #t transform #n #t))
(define train_loader (t_utils_data_DataLoader train_dataset batch_size:128 shuffle:#t num_workers:4 pin_memory:#t))
(define train_itr (t_iter train_loader))
(define test_dataset (tv_datasets_MNIST #f transform #n #t))
(define test_loader (t_utils_data_DataLoader train_dataset batch_size:128 shuffle:#f num_workers:4 pin_memory:#t))
(define test_itr (t_iter test_loader))

(define model
    (t_nn_Sequential (list (t_nn_Conv2d 1 20 5 1)
                           (t_nn_ReLU)
                           (t_nn_MaxPool2d 2 2)
                           (t_nn_Conv2d 20 50 5 1)
                           (t_nn_ReLU)
                           (t_nn_MaxPool2d 2 2)
                           (t_nn_Reshape (list -1 800))
                           (t_nn_Linear 800 500)
                           (t_nn_ReLU)
                           (t_nn_Linear 500 10)
                           (t_nn_LogSoftmax 1))))

(define optimizer (t_optim_SGD (t_parameters model) lr:0.01 momentum:0.9))

(train_test model train_itr test_itr optimizer 100)