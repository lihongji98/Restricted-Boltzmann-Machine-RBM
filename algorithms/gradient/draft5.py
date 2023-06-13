import numpy as np
from rbm import RBM

if __name__ == "__main__":
    data_path = "./data/LS5.txt"
    train_data = np.loadtxt(data_path)
    if train_data.shape[1] == 9:
            initial_path = "BS3/"
    elif train_data.shape[1] == 16:
        initial_path = "BS4/"
    elif train_data.shape[1] == 11:
        initial_path = "LS4/"
    elif train_data.shape[1] == 13:
        initial_path = "LS5/"
    
    dw =  np.load("./gradient/" + initial_path + "exact_dw.npy")
    dvb = np.load("./gradient/" + initial_path + "exact_dvb.npy")
    dhb = np.load("./gradient/" + initial_path + "exact_dhb.npy")

    records = []
    rep_time = 15
    epoch = 100000
    for step in range(rep_time):
        rbm = RBM(v_dim = train_data.shape[1],
                    h_dim = train_data.shape[1] * 3,
                    gibbs_num = 1,
                    opt_type = "cdk",
                    sampling_type = "parallel_tempering", 
                    lr = 0.02,
                    if_lr_decay = True, 
                    epochs= epoch, 
                    batch_size = train_data.shape[0],
                    chain_num = 2,
                    weight_decay = 0, 
                    output_epoch = 10
                    )

        NLL_record = rbm.train(train_data, step)
        records.append(NLL_record)
    records = np.array(records).reshape(rep_time, epoch)

    np.save("records/pt_1e5.npy", records)