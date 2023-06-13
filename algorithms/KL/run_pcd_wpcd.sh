pcd_lr_bs=0.01
pcd_lr_ls=0.05
wpcd_lr_bs=0.03
wpcd_lr_ls=0.05
bs3="./data/BS3.txt"
bs4="./data/BS4.txt"
ls4="./data/LS4.txt"
ls5="./data/LS5.txt" 

python main_train.py --data-path $bs3 --opt-type pcd --lr $pcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $bs4 --opt-type pcd --lr $pcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $ls4 --opt-type pcd --lr $pcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $ls5 --opt-type pcd --lr $pcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 

python main_train.py --data-path $bs3 --opt-type wpcd --lr $wpcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $bs4 --opt-type wpcd --lr $wpcd_lr_bs --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $ls4 --opt-type wpcd --lr $wpcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 
python main_train.py --data-path $ls5 --opt-type wpcd --lr $wpcd_lr_ls --metric_update_epoch 10 --sampling-type "gibbs_sampling" --gibbs-num 1 --if-lr-decay True --weight-decay 0.00025 --epochs 100000 