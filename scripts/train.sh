log_name='train.log'
nohup python train.py \
    --CUDA_VISIBLE_DEVICES 0 \
    --data-dir /home/tmk/project/retinal_classfication/datasets/retionopathy `# 修改为自己的路径` \
    --train-fold 0,1,2,3,4,5,6 `# 共设置了10个fold，可选用其中一部分作为train，其余为val` \
    --batch-size 32 \
    --init-lr 3e-5 \
    --out-dim 7 \
    --n-epochs 15 \
    --num-workers 4 \
    --image-size 224 \
    --DEBUG \
    --load-model \
>$log_name 2>&1 &