log_name='predict.log'
nohup python predict.py \
    --CUDA_VISIBLE_DEVICES 0 \
    --data-dir /home/tmk/project/retinal_classfication/datasets/ `# 修改为自己的路径` \
    --val-fold 6,7,8,9 `# 共设置了10个fold，可选用其中一部分作为train，其余为val` \
    --DEBUG `# DEBUG模式epoch=3，随机选择4个batchsize的数据作为数据集` \
    --batch-size 4 \
    --n-epochs 15 \
    --num-workers 4 \
    --eval best `# 参数文件后缀，best：最好的模型，final：最终模型` \
>$log_name 2>&1 &