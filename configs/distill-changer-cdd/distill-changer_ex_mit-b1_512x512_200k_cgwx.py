_base_ = ['./distill-changer_ex_mit-b0_512x512_200k_cgwx.py']

checkpoint_student = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/Changer-mit-b1/initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_l = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/Changer-mit-b1/initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_s = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/Changer-mit-b1/initial/best_mIoU_iter_1000.pth'

# model settings
model = dict(
    # pretrained=checkpoint,
    
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
    
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))