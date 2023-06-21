import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default= "./algorithms/data/BS3.txt",
        help="paths: [./algorithms/data/BS3.txt, ./algorithms/data/BS4.txt, ./algorithms/data/LS4.txt, ./algorithms/data/LS5.txt]")
    parser.add_argument("--opt-type", type=str, default= "cdk",
        help="choose: [cdk, pcd, wcd, wpcd]")
    parser.add_argument("--sampling-type", type=str, default= "gibbs_sampling",
        help="choose: [gibbs_sampling, parallel_tempering]")
    parser.add_argument("--gibbs-num", type=int, default= 1,
        help="the gibbs sampling time")
    parser.add_argument("--chain-num", type=int, default= 2,
        help="the parallel tempering chain number")
    parser.add_argument("--lr", type=float, default= 1e-3,
        help="the learning rate")
    parser.add_argument("--if-lr-decay", type=bool, default=False,
        help="the linear deacy learning rate")
    parser.add_argument("--weight-decay", type=float, default= 0,
        help="the weight deacy rate")
    parser.add_argument("--epochs", type=int, default= 100000, nargs="?", const=True,
        help="the training epochs")
    parser.add_argument("--metric_update_epoch", type=int, default= 10, nargs="?", const=True,
        help="the training epochs")

    args = parser.parse_args()

    return args