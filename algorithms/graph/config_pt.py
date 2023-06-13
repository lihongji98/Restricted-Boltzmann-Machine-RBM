import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gibbs-num", type=int, default= 1,
        help="the gibbs sampling time")
    parser.add_argument("--chain-num", type=int, default= 2,
        help="the parallel tempering chain number")
    parser.add_argument("--repetition", type=int, default= 1,
        help="the repetition time")
    args = parser.parse_args()
    return args