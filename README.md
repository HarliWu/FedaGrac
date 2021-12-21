# From Deterioration to Acceleration: A Calibration Approach to Rehabilitating Step Asynchronism in Federated Optimization

## Requirements

```
pip install -r requirements.txt
```

For those who target to build on a cluster with OpenMPI, please refer to this blog. Briefly speaking, user should compile Pytorch 1.4.0 from scratch on CUDA-supported OpenMPI. 

## Training

To run the code, you can follow the sample below: 

```
cd federated_learning
python -u start.py --num-workers 100 --partial True --num-part 20 --lr 0.05 --method FedaGrac --lam 0.03 --root ~/dataset --model AlexNet --dataset cifar10  --bsz 25 --non-iid True --dirichlet True --dir-alpha 0.1 --step-async True --step-dist gaussian --inconsistent True --K 500 --variance 100 --T 80
```

Note that ```--lam``` is a hyper-parameter only for FedaGrac. In addition to our proposed method, this repository includes the baselines such as FedNova, SCAFFOLD, FedProx, FedAvg. 

## Citation 

```
@misc{wu2021deterioration,
    title={From Deterioration to Acceleration: A Calibration Approach to Rehabilitating Step Asynchronism in Federated Optimization}, 
    author={Feijie Wu and Song Guo and Haozhao Wang and Zhihao Qu and Haobo Zhang and Jie Zhang and Ziming Liu},
    year={2021}
}
```
