python3 main.py --opt-type cdk --gibbs-num 1 --lr 0.001 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;
cd gradient
rm dw.npy dvb.npy dhb.npy
cd ..

python3 main.py --opt-type cdk --gibbs-num 10 --lr 0.001 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;
cd gradient
rm dw.npy dvb.npy dhb.npy
cd ..

python3 main.py --opt-type wcd --gibbs-num 1 --lr 0.1 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;
cd gradient
rm dw.npy dvb.npy dhb.npy
cd ..

python3 main.py --opt-type pcd --gibbs-num 1 --lr 0.001 --lr-expdecay 1e-11 --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;
cd gradient
rm dw.npy dvb.npy dhb.npy
cd ..

python3 main.py --opt-type wpcd --gibbs-num 1 --lr 0.01 --lr-expdecay 1e-11 --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time;
cd gradient
rm dw.npy dvb.npy dhb.npy
cd ..

# python3 main.py --opt-type exact_gradient --lr 0.01 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;



# lr=(1e-2 1e-3 1e-4)
# h_dim=(9 18 27 36 45)


# python3 main.py --opt-type cdk --gibbs-num 1 --lr 0.001 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# python3 main.py --opt-type cdk --gibbs-num 10 --lr 0.001 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# python3 main.py --opt-type wcd --gibbs-num 1 --lr 0.1 --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# python3 main.py --opt-type pcd --gibbs-num 1 --lr 0.001 --lr-expdecay 1e-11 --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# python3 main.py --opt-type wpcd --gibbs-num 1 --lr 0.001 --lr-expdecay 1e-11 --weight-decay 2.5e-5 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# python3 main.py --opt-type exact_gradient --lr 0.001 --batch-size 14 --hidden-unit 27 --epochs 1000000 --run-time 1;

# # cd1
# for(( i=0;i<${#lr[@]};i++)) do
# 	for(( j=0;j<${#h_dim[@]};j++)) do
#     	python3 main.py --opt-type cdk --gibbs-num 10 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit ${h_dim[j]} --epochs 1000000 --run-time 3;
# 	done;
# done;

# # cd10
# for(( i=0;i<${#lr[@]};i++)) do
# 	for(( j=0;j<${#h_dim[@]};j++)) do
#     	python3 main.py --opt-type cdk --gibbs-num 10 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit ${h_dim[j]} --epochs 300000 --run-time 5;
# 	done;
# done;

# # wcd1
# for(( i=0;i<${#lr[@]};i++)) do
# 	for(( j=0;j<${#h_dim[@]};j++)) do
#     	python3 main.py --opt-type wcd --gibbs-num 1 --lr ${lr[i]} --lr-expdecay 0 --weight-decay 0 --batch-size 14 --hidden-unit ${h_dim[j]} --epochs 300000 --run-time 5;
#     done;
# done;

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