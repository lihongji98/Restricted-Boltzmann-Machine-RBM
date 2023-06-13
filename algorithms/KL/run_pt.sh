pt_lr_bs3=0.005
pt_lr_bs4=0.005
pt_lr_ls=0.02
bs3="./data/BS3.txt"
bs4="./data/BS4.txt"
ls4="./data/LS4.txt"
ls5="./data/LS5.txt" 

python main_train.py --data-path $bs3 --opt-type cdk --lr $pt_lr_bs3 --metric_update_epoch 10 --sampling-type "parallel_tempering" --chain-num 2 --gibbs-num 1 --if-lr-decay True --epochs 100000 
python main_train.py --data-path $bs4 --opt-type cdk --lr $pt_lr_bs4 --metric_update_epoch 10 --sampling-type "parallel_tempering" --chain-num 2 --gibbs-num 1 --if-lr-decay True --epochs 100000 
python main_train.py --data-path $ls4 --opt-type cdk --lr $pt_lr_ls --metric_update_epoch 10 --sampling-type "parallel_tempering" --chain-num 2 --gibbs-num 1 --if-lr-decay True --epochs 100000 
python main_train.py --data-path $ls5 --opt-type cdk --lr $pt_lr_ls --metric_update_epoch 10 --sampling-type "parallel_tempering" --chain-num 2 --gibbs-num 1 --if-lr-decay True --epochs 100000 