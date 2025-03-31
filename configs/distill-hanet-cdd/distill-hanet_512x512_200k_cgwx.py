_base_ = [
    '../_base_/models/Self-KD-hanet.py',
    '../common/standard_512x512_200k_cgwx-cdd.py']

dataset_type = 'LEVIR_CD_Dataset'
data_root = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/ChangeDetectionDataset/ChangeDetectionDataset/Real/subset'

checkpoint_student = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/HANet/initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_l = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/HANet/initial/best_mIoU_iter_1000.pth'
checkpoint_teacher_s = '/nas/datasets/lzy/RS-ChangeDetection/checkpoints_self_distill/CDD/HANet/initial/best_mIoU_iter_1000.pth'

model = dict(
    # student
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s)
)