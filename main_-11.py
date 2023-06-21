import numpy as np
from config import parse_args
from rbm_pn import RBM

if __name__ == "__main__":
    args = parse_args()
    print("*"*50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("*"*50)
    train_data = np.loadtxt(args.data_path)
    if train_data.shape[1] == 9:
            initial_path = "BS3/"
    elif train_data.shape[1] == 16:
        initial_path = "BS4/"
    elif train_data.shape[1] == 11:
        initial_path = "LS4/"
    elif train_data.shape[1] == 13:
        initial_path = "LS5/"

    rbm = RBM(v_dim = train_data.shape[1],
                h_dim = train_data.shape[1] * 3,
                gibbs_num = args.gibbs_num,
                opt_type = args.opt_type,
                sampling_type = args.sampling_type, 
                lr = args.lr,
                if_lr_decay = args.if_lr_decay, 
                epochs= args.epochs, 
                batch_size = train_data.shape[0],
                chain_num = args.chain_num,
                weight_decay = args.weight_decay, 
                output_epoch = args.metric_update_epoch
                )
        
        # rbm.W = np.load('initial/'+initial_path+'W.npy')
        # rbm.v_bias = np.load('initial/'+initial_path+'v_bias.npy')
        # rbm.h_bias = np.load('initial/'+initial_path+'h_bias.npy')

    rbm.train(train_data)