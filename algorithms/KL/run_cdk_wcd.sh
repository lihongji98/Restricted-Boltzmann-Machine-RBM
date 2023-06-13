cdk_lr_bs=0.005
cdk_lr_ls=0.03
wcd_lr_bs=0.1
wcd_lr_ls=0.2
bs3="./data/BS3.txt"
bs4="./data/BS4.txt"
ls4="./data/LS4.txt"
ls5="./data/LS5.txt" 

python main_train.py --data-path $bs3 --opt-type cdk --lr $cdk_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $bs4 --opt-type cdk --lr $cdk_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $ls4 --opt-type cdk --lr $cdk_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $ls5 --opt-type cdk --lr $cdk_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 

python main_train.py --data-path $bs3 --opt-type wcd --lr $wcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $bs4 --opt-type wcd --lr $wcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $ls4 --opt-type wcd --lr $wcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 
python main_train.py --data-path $ls5 --opt-type wcd --lr $wcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --epochs 100000 