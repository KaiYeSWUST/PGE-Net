import sys
import os

sys.path.insert(0, os.getcwd())

custom_imports = dict(
    imports=['projects.gf2_water.loading_gdal'],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


randomness = dict(
    seed=2026,
    deterministic=False,
    diff_rank_seed=False
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=False,
    #影像波段统计值计算，可以使用J:\YEKAI_project_code\HUAHU_Water_SEGFORMER\tools\image_std_value_cal.py工具计算
    mean=[27.27, 20.267, 15.328, 60.288],
    std=[8.251, 6.722, 5.916, 16.695],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=4,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),
    decode_head=dict(
        num_classes=2,
        ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256))
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

dataset_type = 'BaseSegDataset'
data_root = r'J:\YEKAI_project_code\HUAHU_Water_SEGFORMER\data\GID5'

classes = ('background', 'water')
palette = [[0, 0, 0], [0, 0, 255]]

train_pipeline = [
    dict(type='LoadImageFromGDAL', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, gt_seg_map=255)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromGDAL', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, gt_seg_map=255)),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='masks/train'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        metainfo=dict(classes=classes, palette=palette),
        reduce_zero_label=False,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='masks/val'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        metainfo=dict(classes=classes, palette=palette),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

max_iters = 80000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2000)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=20,
        save_best='mIoU',
        rule='greater',
        save_last=True
    ),
    logger=dict(type='LoggerHook', interval=100),
    timer=dict(type='IterTimerHook'),
)

#RUN：python tools/train.py configs/segformer/gid5_segformer_mit_b0_qiepian_base.py
