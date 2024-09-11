

#model_name='ShuffleNetV2'
#model_name='ResNet18'
model_name='ResNet50'
batch_size=128 # 128

#dataset='aug-cifar100'
#data_path='/home/anilkag/code/data/cifar/'
#data_path='/home/anil/github/datasets/cifar/'

dataset='aug-tiny-imagenet-200'
#data_path='/home/anil/github/datasets/tiny-imagenet-200/'
data_path='/Data/rzhu/tiny-imagenet-200/'




CUDA_VISIBLE_DEVICES='2' python train_dyn.py --dataset $dataset --data_path $data_path --model_name $model_name --method 'CE-decoupled' \
	--epochs 200 --lr 0.02 --wd 0.0005 --batch_size 16 --eval_batch_size 128 --ema-decay 0.5 \
	--KD_alpha 0.9 --KD_temperature 3.5 --loss_type 'KLS-MinEnt-Ft12' --reg_ft 2.0 \
       	--lmbda_adaptive 5  --Ti 200 --lmbda 2.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.02 \
	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --budget_g_ft 0.0 --budget_g_stable 0.02





