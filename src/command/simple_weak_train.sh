CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py ctdet --dataset=coco80 --arch dlav0camsplit_34 --exp_id camsplit_weak_train --weak --lr_step 30 --num_epochs 50