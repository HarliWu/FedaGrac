import time, os, json, time
import numpy as np

import torch
from torch._C import device
import torch.distributed as dist
from torch.autograd import Variable

def test_model(model, test_data, dev):
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            output = model(data)
            # get the index of the max log-probability
            _, predictions = output.max(1)
            total += predictions.size(0)
            correct += torch.sum(predictions == target.data).float()

    acc = correct / total
    return acc.item()

def update_model(model, global_mu, size, cpu, gpu):
    # all_param = model.state_dict()

    # receive the parameter from workers 
    for param in model.parameters():
        tensor = torch.zeros_like(param.data, device=cpu)
        gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
        dist.gather(tensor=tensor, gather_list=gather_list, dst=0)
        param.data = torch.zeros_like(param.data, device=gpu)
        for w in range(size):
            # Suppose the model received from clients are well processed 
            param.data = param.data + gather_list[w].clone().detach().to(gpu)

    # receive the mu from clients
    for idx, param in enumerate(global_mu):
        tensor = torch.zeros_like(param.data, device=cpu)
        gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
        dist.gather(tensor=tensor, gather_list=gather_list, dst=0)
        global_mu[idx] = torch.zeros_like(param.data, device=gpu)
        for w in range(size):
            # Suppose the model received from clients are well processed 
            global_mu[idx] = global_mu[idx] + gather_list[w].clone().detach().to(gpu)

    # send the parameters to workers 
    for param in model.parameters():
        tmp_p = param.clone().detach().to(cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)
        if torch.sum(torch.isnan(tmp_p)) > 0:
            print("NaN occurs. Terminate. ")
            exit(-1)

    # send global_mu to workers
    for param in global_mu:
        tmp_p = param.clone().detach().to(cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)

    # model.load_state_dict(all_param)

def run(size, model, args, test_data, f_result, cpu, gpu):
    # Receive the weights from all clients 
    temp_w = torch.tensor([0.0 for _ in range(args.num_workers+1)])
    weights = [torch.tensor([0.0 for _ in range(args.num_workers+1)]) for _ in range(size)]
    dist.gather(tensor=temp_w, gather_list=weights, dst=0)
    weights = sum(weights)
    weights = weights / torch.sum(weights)
    print('weights:', weights)

    # send weights to clients
    weights_list = [weights.clone().detach().to(cpu) for _ in range(size)]
    dist.scatter(tensor=temp_w, scatter_list=weights_list)
    
    start = time.time()
    model = model.cuda(gpu)

    for p in model.parameters():
        tmp_p = p.clone().detach().to(cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        # dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list, group=group)
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)

    global_mu = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]

    print('Model has sent to all nodes! ')
    print('Begin!') 

    np.random.seed(42)

    for t in range(args.T):
        model.train()
        # send participants to all clients 
        participants = np.random.choice(np.arange(len(weights)), size=args.num_part, replace=True, p=weights.numpy()) if args.partial else np.arange(len(weights))
        print('Participants list:', list(participants))
        participants = torch.tensor(participants).to(cpu)
        part_list = [participants for _ in range(size)]
        dist.scatter(tensor=participants, scatter_list=part_list)

        # receive the list of train loss from workers
        info_list = [torch.tensor(0.0) for _ in range(size)]
        # dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, group=group)
        dist.gather(tensor=torch.tensor(0.0), gather_list=info_list, dst=0)
        # info_list = np.concatenate([list(a) for a in info_list])
        # train_loss = sum(info_list).item() / args.num_part if args.partial else sum(info_list * weights).item()
        train_loss = sum(info_list).item()

        # if args.partial:
        #     update_model_partial(model, size, cpu, gpu, args.num_part)
        # else:
        #     update_model_full(model, size, cpu, gpu, weights)
        update_model(model, global_mu, size, cpu, gpu)

        timestamp = time.time() - start
        test_acc = test_model(model, test_data, gpu)
        print("Epoch: {}\t\tLoss: {}\t\tAccuracy: {}".format(t, train_loss, test_acc))
        f_result.write(str(t) + "\t" + str(timestamp) + "\t" + str(train_loss) + "\t" + str(test_acc) + "\n")
        f_result.flush()

def init_processes(rank, size, model, args, test_data, cpu, gpu, backend='mpi'):
    if backend == 'mpi':
        dist.init_process_group(backend)
    elif backend == 'gloo':
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    result_file = os.path.join(args.result, '{}.txt'.format(len(os.listdir(args.result))))
    f_result = open(result_file, 'w')
    f_result.write(json.dumps(vars(args)) + '\n')
    run(size, model, args, test_data, f_result, cpu, gpu)
