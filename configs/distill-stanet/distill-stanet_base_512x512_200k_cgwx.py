_base_ = [
    '../_base_/models/Self-KD-stanet_r18.py',
    '../common/standard_512x512_200k_cgwx.py']

crop_size = (256, 256)

checkpoint_student = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/STANet/Initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_l = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/STANet/Initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_s = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/STANet/Initial/best_mIoU_iter_1000.pth'

model = dict(
    # student
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),

    decode_head=dict(sa_mode='None'),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=200000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 100k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=200000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, 
                       img_shape=(512, 512, 3)))