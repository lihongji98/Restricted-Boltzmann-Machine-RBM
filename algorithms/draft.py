import numpy as np
from config import parse_args
from cdk import RBM as cdk
from pcd import RBM as pcd
from wcd import RBM as wcd
from wpcd import RBM as wpcd
from exact_gradient import RBM as exact_gradient

if __name__ == "__main__":
    # args = parse_args()
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")

    # if args.opt_type == "cdk":
    #     model = cdk
    # elif args.opt_type == "pcd":
    #     model = pcd
    # elif args.opt_type == "wcd":
    #     model = wcd
    # elif args.opt_type == "wpcd":
    #     model = wpcd
    # elif args.opt_type == "exact_gradient":
    #     model = exact_gradient
    # else:
    #     print("choose a correct optimization method")

    # ss = np.random.normal(-1, 1, size = (1000000,))
    # np.save(f"./distance/cos_dw_{args.opt_type}.npy", ss)
    v_bias = np.random.normal(-0.1, 0.1, size = (1, 9))
    h_bias = np.random.normal(-0.1, 0.1, size =(1, 27))
    W = np.random.normal(size = (9, 27))
    np.save("W_3.npy", W)
    np.save("v_bias_3.npy", v_bias)
    np.save("h_bias_3.npy", h_bias)