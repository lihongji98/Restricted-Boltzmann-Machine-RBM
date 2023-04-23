#lr=(1e-2 3e-3 2.5e-3 2e-3 1e-3 1e-4)
lr=(1e-1 5e-1 3e-1)
pc_lr=(1e-2 2.5e-3 1e-3 1e-4)
pc_lr_decay=(1e-9 1e-10 1e-11)

# # cd1
# for(( i=0;i<${#lr[@]};i++)) do
#     python3 main.py --opt-type cdk --gibbs-num 1 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 20 --epochs 300000 --run-time 5;
# done;

# # cd10
# for(( i=0;i<${#lr[@]};i++)) do
#     python3 main.py --opt-type cdk --gibbs-num 10 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 20 --epochs 300000 --run-time 5;
# done;

# wcd1
for(( i=0;i<${#lr[@]};i++)) do
    python3 main.py --opt-type wcd --gibbs-num 1 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 20 --epochs 300000 --run-time 5;
done;

# # pcd1
# for(( i=0;i<${#pc_lr[@]};i++)) do
# 	for(( j=0;j<${#pc_lr_decay[@]};j++)) do
# 		python3 main.py --opt-type pcd --gibbs-num 1 --lr ${pc_lr[i]} --lr-expdecay ${pc_lr_decay[j]} --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 20 --epochs 300000 --run-time 5;
# 	done;
# done;

# # wpcd1
# for(( i=0;i<${#pc_lr[@]};i++)) do
# 	for(( j=0;j<${#pc_lr_decay[@]};j++)) do
# 		python3 main.py --opt-type wpcd --gibbs-num 1 --lr ${pc_lr[i]} --lr-expdecay ${pc_lr_decay[j]} --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 20 --epochs 300000 --run-time 5;
# 	done;
# done;