import numpy as np
from rbm import RBM
from config_pt import parse_args

if __name__ == "__main__":
    args = parse_args()
    data_path = "./data/LS4.txt"
    opt_name = "cdk"
    opt_name_store = "pt"
    opt_lr = 0.003
    opt_lr_decay = True
    opt_sampling_type = "parallel_tempering"

    print("*"*50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("*"*50)

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
    rep_time = args.repetition
    epoch = 100000
    for step in range(rep_time):
        rbm = RBM(v_dim = train_data.shape[1],
                    h_dim = train_data.shape[1] * 3,
                    gibbs_num = args.gibbs_num,
                    chain_num = args.chain_num,
                    opt_type = opt_name,
                    sampling_type = opt_sampling_type, 
                    lr = opt_lr,
                    if_lr_decay = opt_lr_decay, 
                    epochs= epoch, 
                    batch_size = train_data.shape[0],
                    weight_decay = 0, 
                    output_epoch = 100
                    )

        optimal_record = rbm.train(train_data, step)
        KL.append(optimal_record)
    KL = np.array(KL).reshape(rep_time)
    output = np.mean(KL,axis=0)
    print(output)