import numpy as np
from rbm import RBM
from norm import norm_compute

if __name__ == "__main__":
    data_path = "./data/LS5.txt"
    opt_name = "cdk"
    opt_name_store = "pt"
    opt_lr = 0.02
    opt_lr_decay = True
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
    
    dw =  np.load("./gradient/" + initial_path + opt_name + "_dw.npy")
    dvb = np.load("./gradient/" + initial_path + opt_name + "_dvb.npy")
    dhb = np.load("./gradient/" + initial_path + opt_name + "_dhb.npy")

    norm_computation = norm_compute(name = "RBM")

    dw_records, dvb_records, dhb_records = [], [], []
    rep_time = 20
    epoch = 3000
    for step in range(rep_time):
        rbm = RBM(v_dim = train_data.shape[1],
                    h_dim = train_data.shape[1] * 3,
                    gibbs_num = 1,
                    opt_type = opt_name,
                    sampling_type = opt_sampling_type, 
                    lr = opt_lr,
                    if_lr_decay = opt_lr_decay, 
                    epochs= epoch, 
                    batch_size = train_data.shape[0],
                    chain_num = 2,
                    weight_decay = 0, 
                    output_epoch = 1
                    )

        dw_approxiamtion, dvb_approxiamtion, dhb_approxiamtion = rbm.train(train_data, step)
        dw_approxiamtion = np.array(dw_approxiamtion).reshape(epoch, train_data.shape[1], train_data.shape[1] * 3)
        dvb_approxiamtion = np.array(dvb_approxiamtion).reshape(epoch, 1, train_data.shape[1])
        dhb_approxiamtion = np.array(dhb_approxiamtion).reshape(epoch, 1, train_data.shape[1] * 3)

        dw_difference, dvb_difference, dhb_difference = norm_computation.cos_sim(dw, dvb, dhb, dw_approxiamtion, dvb_approxiamtion, dhb_approxiamtion)
        
        dw_records.append(dw_difference)
        dvb_records.append(dvb_difference)
        dhb_records.append(dhb_difference)
    
    dw_records = np.array(dw_records).reshape(rep_time, epoch)
    dvb_records = np.array(dvb_records).reshape(rep_time, epoch)
    dhb_records = np.array(dhb_records).reshape(rep_time, epoch)

    dw_records = np.mean(dw_records, axis=0)
    dvb_records = np.mean(dvb_records, axis=0)
    dhb_records = np.mean(dhb_records, axis=0)

    np.save("distance/LS5/" + opt_name_store + "_dw.npy", dw_records)
    np.save("distance/LS5/" + opt_name_store + "_dvb.npy", dvb_records)
    np.save("distance/LS5/" + opt_name_store + "_dhb.npy", dhb_records)