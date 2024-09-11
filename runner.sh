CUDA_VISIBLE_DEVICES='0' python train_dyn.py --dataset 'aug-cifar100' --model_name 'ResNet18' --method 'DCL' \
	--epochs 200 --lr 0.1 --lr_t 0.1 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.6 \
	--KD_alpha 1.0 	--KD_temperature 4  --reg_ft 2.0 --lmbda_adaptive 5  --Ti 50 --lmbda 5.0 --lmbda_min .05 --lmbda_dual 0.0 --budget_g 0.1 \
	--lmbda_dual_ent 0.0 --budget_g_ent 0.3 --lmbda_max_ent 1.0 --interval=1 --rand_seed 4