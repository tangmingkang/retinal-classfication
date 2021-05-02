log_name='visual.log'
nohup python grad_cam.py \
    --root-path /home/tmk/project/retinal_classfication/ `# 修改为自己的路径` \
    --image-path datasets/image/000326720200314010002.jpg `# 需要可视化的图像` \
    --kernel_type efficientnet_b3_size512_outdim2_bs4_best.pth `# 需要可视化的模型` \
>$log_name 2>&1 &