import numpy as np
from config import parse_args
from cdk import RBM as cdk
from pcd import RBM as pcd
from wcd import RBM as wcd
from wpcd import RBM as wpcd



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
    else:
        print("choose a correct optimization method")


    train_data = np.loadtxt(r'../3x3.txt')
    print("*"*50 + args.opt_type + "-" + str(args.gibbs_num) + "*"*50)
    for i in range(args.run_time):
        rbm = model(v_dim = train_data.shape[1],
                    h_dim = args.hidden_unit,
                    lr = args.lr,
                    exp_lrd = args.lr_expdecay,
                    weight_decay = args.weight_decay,
                    epochs= args.epochs ,
                    batch_size = args.batch_size,
                    gibbs_num = args.gibbs_num,
                    )

        rbm.train(train_data)
    print("*"*50)
