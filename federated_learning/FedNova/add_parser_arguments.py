from argparse import ArgumentParser

def new_arguements(parser: ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr-diminish', type=bool, default=False, help='diminish the learning rate')
    parser.add_argument('--lr-decay', type=int, default=10, help='number of communication rounds diminish the learning rate')
    parser.add_argument('--T', type=int, default=100, help='Communication rounds')

    # Setting for local iterations 
    parser.add_argument('--K', type=int, default=50, help='Local iterations')
    parser.add_argument('--step-async', type=bool, default=False, help='Step Asynchronism')
    parser.add_argument('--step-dist', type=str, default='gaussian', help='(epoch_)uniform, linear, gaussian, extreme')
    parser.add_argument('--variance', type=float, default=100.0, help='variance for gaussian')  
    parser.add_argument('--k-min', type=int, default=100, help='minimum number of local iterations')
    parser.add_argument('--k-max', type=int, default=100, help='maximum number of local iterations')
    parser.add_argument('--inconsistent', type=bool, default=False, help='numbers of local updates vary among communication rounds')
