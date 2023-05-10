import argparse

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-type", type=str, default= "cdk",
        help="the type of optimization")
    parser.add_argument("--gibbs-num", type=int, default= 1,
        help="the gibbs sampling time")
    parser.add_argument("--batch-size", type=int, default= 14, nargs="?", const=True,
        help="the batch size")
    parser.add_argument("--lr", type=float, default= 1e-3,
        help="the learning rate")
    parser.add_argument("--lr-expdecay", type=float, default= 0, nargs="?", const=True,
        help="the exponential learning decay rate")
    parser.add_argument("--weight-decay", type=float, default= 0, nargs="?", const=True,
        help="the weight deacy rate")
    parser.add_argument("--hidden-unit", type=int, default= 27, nargs="?", const=True,
        help="the number of hidden units")
    parser.add_argument("--epochs", type=int, default= 300000, nargs="?", const=True,
        help="the training epochs")
    parser.add_argument("--run-time", type=int, default= 1, nargs="?", const=True,
        help="the running time")

    args = parser.parse_args()

    return args