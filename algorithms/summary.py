from config import parse_args
import gym
import time
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs

if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    names = ["dw","dvb", "dhb"]
    opt_types = ["cdk", "cdk10", "wcd", "pcd", "wpcd"]
    sim_types = ["cos", "F"]
    num_ = 3

    def record_d(sim_type, opt_type, epochs):
        writer = SummaryWriter(f"runs/{sim_type}__{opt_type}_{num_}")

        # dw = np.load(f"./distance/{opt_type}_{sim_type}_dw_{num_}.npy")
        # for i in tqdm(range(epochs)):
        #     if sim_type == "cos":
        #         writer.add_scalar(f"cos_sim/dw_{num_}_{epochs}",dw[i], i + 1)
        #     else:
        #         writer.add_scalar(f"F_norm/dw_{num_}_{epochs}", dw[i], i + 1)

        dhb = np.load(f"./distance/{opt_type}_{sim_type}_dhb_{num_}.npy")
        for i in tqdm(range(epochs)):
            if sim_type == "cos":
                writer.add_scalar(f"cos_sim/dhb_{num_}_{epochs}",dhb[i], i + 1)
            else:
                writer.add_scalar(f"F_norm/dhb_{num_}_{epochs}", dhb[i], i + 1)

        dvb = np.load(f"./distance/{opt_type}_{sim_type}_dvb_{num_}.npy")
        for i in tqdm(range(epochs)):
            if sim_type == "cos":
                writer.add_scalar(f"cos_sim/dvb_{num_}_{epochs}",dvb[i], i + 1)
            else:
                writer.add_scalar(f"F_norm/dvb_{num_}_{epochs}", dvb[i], i + 1)
        writer.close()



    # for i in range(len(sim_types)):
    #     for j in range(len(opt_types)):
    #         record_dhb(sim_types[i], opt_types[j])
    epochs = 100000
    record_d(sim_types[0], opt_types[2], epochs) # wcd
    record_d(sim_types[0], opt_types[4], epochs) # wpcd
    record_d(sim_types[0], opt_types[3], epochs) # pcd
    record_d(sim_types[0], opt_types[0], epochs) # cd1














