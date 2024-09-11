
#data_path='/home/anilkag/code/data/cifar/'
dataset='aug-cifar100'
#data_path='/home/anil/github/datasets/cifar/'

dataset='aug-tiny-imagenet-200'
data_path='/home/anil/github/datasets/tiny-imagenet-200/'
ckpt='"'

CUDA_VISIBLE_DEVICES='2' python train.py --dataset $dataset  --data_path $data_path --model_name 'ResNet50' --method 'CE' \
	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 92 --eval_batch_size 128

#CUDA_VISIBLE_DEVICES='2' python train.py --dataset $dataset  --data_path $data_path --model_name 'ShuffleNetV2' --method 'CE' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 512


#CUDA_VISIBLE_DEVICES='0' python train.py --dataset 'aug-cifar100' --data_path $data_path  --model_name 'ResNet50' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft2' --reg_ft 2.0 \
#       	--lmbda_adaptive 5  --Ti 200 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.1 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt 


#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#CUDA_VISIBLE_DEVICES='0' python train.py --dataset 'aug-cifar100' --data_path $data_path  --model_name 'ResNet50' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 92 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft9' --reg_ft 1.0 \
#       	--lmbda_adaptive 5  --Ti 200 --lmbda 10.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.05 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt --budget_g_ft 0.05 --budget_g_stable 0.05 


#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'cifar100' --model_name 'ResNet18' --method 'SD' \
#	--epochs 200 --lr 0.1 --wd 0.00001 --batch_size 256 --eval_batch_size 512 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0  

#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.00001 --batch_size 256 --eval_batch_size 512 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0  

# 81.02%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0  

# 78.87%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0  

# 81.18%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0 --loss_type 'KLS' 

# 80.09%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0 --loss_type 'MSE' 


#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0 --loss_type 'MSES' 

# 79.79%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'CE' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.5 --KD_temperature 4.0  

# 82.02%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 4.0 --loss_type 'KLS' \
#       	--lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --budget_g 0.0

# 82.08%
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.0 --loss_type 'KLS' \
#       	--lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --budget_g 0.0

# 81.43% Neely's algorithm
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.0 --loss_type 'KLS' \
#       	--lmbda_adaptive 6  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.0


# 82.31% Neely's algorithm
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS' \
#       	--lmbda_adaptive 6  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.0

#ckpt='./gold_models/aug-cifar100-our-SD-ResNet18-KLS-MinEnt-200-5-50-5.0-0.05-0.0-0.4---model_best.pth.tar'
#ckpt='./gold_models/aug-cifar100-our-SD-ResNet18-KLS-200-5-50-5.0-0.05---model_best.pth.tar'
#ckpt='./gold_models/aug-cifar100-our-SD-ResNet18-KLS-200-6-50-5.0-0.05---model_best.pth.tar'
#ckpt='./gold_models/aug-cifar100-CE-ResNet18-CE-200---model_best.pth.tar'
ckpt='"'

# 82.05% Neely's algorithm ( Min-Entropy + KLS )
#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt' \
#       	--lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.01 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt 


#CUDA_VISIBLE_DEVICES='2' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft' \
#       	--lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.1 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt 


#CUDA_VISIBLE_DEVICES='0' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft2' --reg_ft 0.8 \
#       	--lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.1 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt 
#

#CUDA_VISIBLE_DEVICES='0' python train.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'our-SD' \
#	--epochs 200 --lr 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
#	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft3' --reg_ft 0.8 \
#       	--lmbda_adaptive 5  --Ti 200 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.1 \
#	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --_ckpt $ckpt 




#
#	--KD_alpha 0.5 --KD_temperature 4.0 --loss_type 'KLS' \
