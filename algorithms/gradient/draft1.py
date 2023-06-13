import numpy as np
from rbm import RBM

if __name__ == "__main__":
    data_path = "./data/LS5.txt"
    opt_name = "cdk"
    opt_name_store = "pt"
    opt_lr = 0.005
    opt_lr_decay = False
    opt_sampling_type = "parallel_tempering"

    train_data = np.loadtxt(data_path)
    if train_data.shape[1] == 9:
        initial_path = "BS3/"
    elif train_data.shape[1] == 16:
        initial_path = "BS4/"
    elif train_data.shape[1] == 11:
        initial_path = "LS4/"
    elif train_data.shape[1] == 13:
        initial_path = "LS5/"

    KL = []
    rep_time = 1
    epoch = 100000
    for step in range(rep_time):
        rbm = RBM(v_dim = train_data.shape[1],
                    h_dim = train_data.shape[1] * 3,
                    gibbs_num = 10,
                    opt_type = opt_name,
                    sampling_type = opt_sampling_type, 
                    lr = opt_lr,
                    if_lr_decay = opt_lr_decay, 
                    epochs= epoch, 
                    batch_size = train_data.shape[0],
                    chain_num = 2,
                    weight_decay = 0, 
                    output_epoch = 500
                    )

        KL_records = rbm.train(train_data, step)
        KL_records = np.array(KL_records).reshape(epoch)

        # KL.append(KL_records)

    
    # KL = np.array(KL).reshape(rep_time, epoch)
    # KL = np.mean(KL, axis=0)
    # np.save("KL/BS4/" + opt_name_store + "_1e3.npy", KL)