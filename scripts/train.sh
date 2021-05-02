log_name='train.log'
nohup python train.py \
    --CUDA_VISIBLE_DEVICES 0 \
    --data-dir /home/tmk/project/retinal_classfication/datasets/ `# 修改为自己的路径` \
    --train-fold 0,1,2,3,4,5 `# 共设置了10个fold，可选用其中一部分作为train，其余为val` \
    --DEBUG `# DEBUG模式epoch=3，随机选择4个batchsize的数据作为数据集` \
    --batch-size 4 \
    --init-lr 3e-5 \
    --n-epochs 15 \
    --num-workers 4 \
>$log_name 2>&1 &