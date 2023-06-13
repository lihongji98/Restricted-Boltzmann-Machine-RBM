import numpy as np
from config import parse_args
from cdk import RBM as cdk
from pcd import RBM as pcd
from wcd import RBM as wcd
from wpcd import RBM as wpcd
from exact_gradient import RBM as exact_gradient

from norm import norm_compute

if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.opt_type == "cdk":
        model = cdk
    elif args.opt_type == "pcd":
        model = pcd
    elif args.opt_type == "wcd":
        model = wcd
    elif args.opt_type == "wpcd":
        model = wpcd
    elif args.opt_type == "exact_gradient":
        model = exact_gradient
    else:
        print("choose a correct optimization method")


    train_data = np.loadtxt(r'./data/Labeled-Shifter-5-13.txt')
                            # './data/Bars-and-Stripes-3x3.txt' (14, 9) 27
                            # './data/Bars-and-Stripes-4x4.txt' (30, 16) 48
                            # './data/Labeled-Shifter-4-11.txt' (48, 11) 33
                            # './data/Labeled-Shifter-5-13.txt' (96, 13) 39
    print("*"*50 + args.opt_type + "-" + str(args.gibbs_num) + "*"*50)
    for i in range(args.run_time):
        if args.opt_type != "exact_gradient":
            rbm = model(v_dim = train_data.shape[1],
                        h_dim = args.hidden_unit,
                        lr = args.lr,
                        exp_lrd = args.lr_expdecay,
                        weight_decay = args.weight_decay,
                        epochs= args.epochs ,
                        batch_size = args.batch_size,
                        gibbs_num = args.gibbs_num,
                        )
        else:
            rbm = model(v_dim = train_data.shape[1],
                        h_dim = args.hidden_unit,
                        lr = args.lr,
                        epochs= args.epochs ,
                        batch_size = args.batch_size,
                        )
        rbm.W = np.load('initial/W_3.npy')
        rbm.v_bias = np.load('initial/v_bias_3.npy')
        rbm.h_bias = np.load('initial/h_bias_3.npy')

        #rbm.train(train_data)
        rbm.gradient_compare(train_data)
    print("*"*50)



    # norm_ =  norm_compute()
    # exact_dw   = np.load("./gradient/exact_dw.npy")
    # exact_dvb  = np.load("./gradient/exact_dvb.npy")
    # exact_dhb =  np.load("./gradient/exact_dhb.npy")
    # opt_dw     = np.load("./gradient/dw.npy")
    # opt_dvb    = np.load("./gradient/dvb.npy")
    # opt_dhb    = np.load("./gradient/dhb.npy")

    # dw_cos, dvb_cos, dhb_cos = norm_.cos_sim(exact_dw,exact_dvb, exact_dhb, opt_dw, opt_dvb, opt_dhb)
    # dw_F, dvb_F, dhb_F = norm_.Frobenius_norm(exact_dw,exact_dvb, exact_dhb, opt_dw, opt_dvb, opt_dhb)
    # np.save(f"./distance/{args.opt_type}_cos_dw_3.npy", dw_cos)
    # np.save(f"./distance/{args.opt_type}_cos_dvb_3.npy", dvb_cos)
    # np.save(f"./distance/{args.opt_type}_cos_dhb_3.npy", dhb_cos)

    # np.save(f"./distance/{args.opt_type}_F_dw_3.npy", dw_F)
    # np.save(f"./distance/{args.opt_type}_F_dvb_3.npy", dvb_F)
    # np.save(f"./distance/{args.opt_type}_F_dhb_3.npy", dhb_F)












