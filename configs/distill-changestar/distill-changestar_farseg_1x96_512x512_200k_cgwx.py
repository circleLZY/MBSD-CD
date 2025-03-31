_base_ = [
    '../_base_/models/KD-changestar_farseg_1x96_r18.py', 
    '../common/standard_512x512_200k_cgwx.py']

dataset_type = 'LEVIR_CD_Dataset'
data_root = '/nas/datasets/lzy/RS-ChangeDetection/CGWX'

crop_size = (512, 512)

checkpoint_student = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_distill/ChangeStar/teacher_ckpt/initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_l = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_distill/ChangeStar/teacher_ckpt/large/best_mIoU_iter_1000.pth'
checkpoint_teacher_m = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_distill/ChangeStar/teacher_ckpt/medium/best_mIoU_iter_1000.pth'
checkpoint_teacher_s = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_distill/ChangeStar/teacher_ckpt/small/best_mIoU_iter_1000.pth'

train_dataloader = dict(batch_size=16, num_workers=8)

model = dict(
    # student
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher medium    
    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
)

optimizer=dict(
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=100000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 100k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=1000)
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