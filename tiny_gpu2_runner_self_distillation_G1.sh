
#data_path='/home/anilkag/code/data/cifar/'
#dataset='aug-cifar100'
#data_path='/home/anil/github/datasets/cifar/'

dataset='aug-tiny-imagenet-200'
data_path='/home/anil/github/datasets/tiny-imagenet-200/'
ckpt='"'

# CE -- Standard, EMA -- 62.43%, 65.02%
#CUDA_VISIBLE_DEVICES='3' python train.py --dataset $dataset  --data_path $data_path --model_name 'ResNet18' --method 'CE' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 512


#CUDA_VISIBLE_DEVICES='3' python train.py --dataset $dataset  --data_path $data_path --model_name 'ResNet50' --method 'CE' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 32 --eval_batch_size 512


CUDA_VISIBLE_DEVICES='3' python train.py --dataset $dataset --data_path $data_path  --model_name 'ResNet18' --method 'our-SD' \
	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 92 --eval_batch_size 256 --ema-decay 0.996 \
	--KD_alpha 0.9 --KD_temperature 4.0 --loss_type 'KLS-MinEnt-Ft10' --reg_ft 1.0 \
       	--lmbda_adaptive 5  --Ti 200 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.05 \
	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt --budget_g_ft 0.05 --budget_g_stable 0.05 


