import numpy as np
from copy import deepcopy
import os, math

import torch
from torch._C import device
from torch.optim import SGD
import torch.distributed as dist
from torch.autograd import Variable

def update_model(model, aggregated_model, cpu, gpu):
    # all_param = model.state_dict()
    
    # send parameters to PS
    for param in aggregated_model:
        param_cpu = param.clone().detach().to(cpu)
        dist.gather(tensor=param_cpu, dst=0)
    
    # receive parameters from PS
    for param in model.parameters():
        recv = torch.zeros_like(param.data, device=cpu)
        dist.scatter(tensor=recv, src=0)
        if torch.sum(torch.isnan(recv)) > 0:
            exit(-1)
        param.data = recv.clone().detach().to(gpu)

    # model.load_state_dict(all_param)

def get_num_steps(worker_idx, args, data_len):
    if args.step_async is False:
        return args.K
    step_dist = args.step_dist.lower()
    if args.inconsistent is False:
        np.random.seed(42*worker_idx)
        np.random.seed(np.random.randint(12345))
    
    coeff = data_len if "epoch" in step_dist else 1

    if "gaussian" in step_dist:
        return int(abs(np.random.normal(args.K, math.sqrt(args.variance)))) * coeff
    if "linear" in step_dist:
        return (args.k_min + args.K * worker_idx) * coeff 
    if "uniform" in step_dist:
        return np.random.randint(args.k_min, args.k_max) * coeff
    if "extreme" in step_dist:
        return (args.k_max if worker_idx == args.num_workers else args.k_min) * coeff

def run(workers, size, model, args, data_ratio_pairs:dict, cpu, gpu):
    # Send the weights to server 
    weights = torch.tensor([0.0 for _ in range(args.num_workers+1)])
    for worker_id, (_, w) in data_ratio_pairs.items():
        weights[worker_id] = w
    # weights = [w for _, w in data_ratio_pairs.values()]
    dist.gather(tensor=weights.clone().detach().to(cpu), dst=0)

    # receive updated weights from clients 
    dist.scatter(tensor=weights, src=0)

    model = model.cuda(gpu)
    model.train()
    iterators = [iter(train_data) for train_data, _ in data_ratio_pairs.values()]

    # Receive initial model from server
    for idx, p in enumerate(model.parameters()):
        tmp_p = torch.zeros_like(p, device=cpu)
        dist.scatter(tensor=tmp_p, src=0)
        p.data = tmp_p.clone().detach().to(gpu)

    print('Worker {} successfully received the model. '.format(list(workers)))

    for t in range(args.T):
        # Receive participants list 
        part_list = torch.tensor(np.arange(args.num_part)).to(cpu) if args.partial else torch.tensor(np.arange(size)).to(cpu)
        dist.scatter(tensor=part_list, src=0)
        part_list = part_list.numpy()

        aggregated_loss  = torch.tensor(0.0, device=gpu)
        aggregated_model = [torch.zeros_like(param, device=gpu) for param in model.parameters()]

        for idx, worker in enumerate(workers):
            if worker in part_list:
                mymodel, K = deepcopy(model), get_num_steps(worker, args, len(data_ratio_pairs[worker][0]))
                mymodel.train()
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = SGD(mymodel.parameters(), lr=args.lr)
                tot_loss = 0.0

                # perform local update 
                for _ in range(K):
                    try:
                        data, target = next(iterators[idx])
                    except:
                        iterators[idx] = iter(data_ratio_pairs[worker][0])
                        data, target = next(iterators[idx])
                    target = torch.tensor(target).to(gpu) if type(target) == int else target.to(gpu)
                    data = data.to(gpu)
                    # data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
                    optimizer.zero_grad()
                    output = mymodel(data)
                    loss = criterion(output, target)
                    regularization = torch.tensor(0.0, device=gpu)
                    for param_1, param_2 in zip(mymodel.parameters(), model.parameters()):
                        regularization += torch.sum(torch.abs(param_1 - param_2))
                    regularization = regularization ** 2
                    tot_loss = tot_loss + loss.data / K
                    loss += args.mu * regularization
                    loss.backward()
                    optimizer.step()
                
                print('Worker: {}   Communition Rounds: {}    Local Updates: {}    Loss: {}'.format(worker, t, K, tot_loss))
                weight = np.count_nonzero(part_list==worker) / len(part_list) if args.partial else weights[worker]
                aggregated_loss = aggregated_loss + tot_loss * weight
                aggregated_model = [aggregated_model[idx] + (param.data * weight) for idx, param in enumerate(mymodel.parameters())]

        loss_cpu = aggregated_loss.clone().detach().to(cpu)
        dist.gather(tensor=loss_cpu, dst=0)

        update_model(model, aggregated_model, cpu, gpu)

        if args.lr_diminish and (t+1) % args.lr_decay == 0:
            args.lr *= 0.8
        

def init_processes(rank, workers, size, model, args, data_ratio_pairs, cpu, gpu, backend='mpi'):
    if backend == 'mpi':
        dist.init_process_group(backend)
    elif backend == 'gloo':
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
    np.random.seed(42*rank)
    np.random.seed(np.random.randint(12345))
    run(workers, size, model, args, data_ratio_pairs, cpu, gpu)