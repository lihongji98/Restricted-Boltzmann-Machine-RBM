import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# # tensorboard --logdir=runs

if __name__ == "__main__":
    def write_record(gradient_type, dataset):
        writer = SummaryWriter('runs/' + gradient_type)
        cdk = np.load("./distance/" + dataset +"/" + "cdk" + "_" + gradient_type + ".npy")
        pcd = np.load("./distance/" + dataset +"/" + "pcd" + "_" + gradient_type + ".npy")
        wcd = np.load("./distance/" + dataset +"/" + "wcd" + "_" + gradient_type + ".npy")
        wpcd = np.load("./distance/" + dataset +"/" + "wpcd" + "_" + gradient_type + ".npy")
        pt = np.load("./distance/" + dataset +"/" + "pt" + "_" + gradient_type + ".npy")
        for i in range(len(cdk)):
            writer.add_scalars(gradient_type, {'cdk':cdk[i], 
                                        'pcd': pcd[i],
                                        'wcd': wcd[i],
                                        'wpcd': wpcd[i],
                                        'pt': pt[i]},
                                            i+1) 
        writer.close()
    
    write_record("dw", "BS4")
    write_record("dvb", "BS4")
    write_record("dhb", "BS4")


