# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from opencd.registry import MODELS


@MODELS.register_module()
class BAN(BaseSegmentor):
    """A New Learning Paradigm for Foundation Model-based 
    Remote Sensing Change Detection. `BAN <arxiv.org/abs/2312.01163>` _.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train()
     _decode_head_forward_train(): decode_head.loss()

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     inference(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        image_encoder (ConfigType): The config for the visual encoder of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        asymetric_input (bool): whether to use different size of input for image encoder
            and decode head. Defaults to False.
        encoder_resolution (float): resize scale of input images for image encoder.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 image_encoder: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 asymetric_input: bool = True,
                 encoder_resolution: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            image_encoder.init_cfg = dict(
                type='Pretrained_Part', checkpoint=pretrained)
            decode_head.init_cfg = dict(
                type='Pretrained_Part', checkpoint=pretrained)

        if asymetric_input:
            assert encoder_resolution is not None, \
                'if asymetric_input set True, ' \
                'clip_resolution must be a certain value'
        self.asymetric_input = asymetric_input
        self.encoder_resolution = encoder_resolution
        self.image_encoder = MODELS.build(image_encoder)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract visual features from images."""
        x = self.image_encoder(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode the name of classes with text_encoder and encode images with
        image_encoder.

        Then decode the class embedding and visual feature into a semantic
        segmentation map of the same size as input.
        """
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.image_encoder(fm_img_from)
        fm_feat_to = self.image_encoder(fm_img_to)
        seg_logits = self.decode_head.predict([img_from, img_to, fm_feat_from, fm_feat_to],
                                              batch_img_metas, self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.image_encoder(fm_img_from)
        fm_feat_to = self.image_encoder(fm_img_to)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            [img_from, img_to, fm_feat_from, fm_feat_to], data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.extract_feat(fm_img_from)
        fm_feat_to = self.extract_feat(fm_img_to)
        return self.decode_head.forward([img_from, img_to, fm_feat_from, fm_feat_to])

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    
@MODELS.register_module()
class DistillBAN(BAN):
    def __init__(self,
                 distill_loss,
                 image_encoder: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 asymetric_input: bool = True,
                 encoder_resolution: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        super().__init__(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg)

        self.teacher_l = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_l)
        
        self.teacher_m = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_m)
        
        self.teacher_s = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_s)
        
        self.distill_loss = MODELS.build(distill_loss)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.image_encoder(fm_img_from)
        fm_feat_to = self.image_encoder(fm_img_to)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            [img_from, img_to, fm_feat_from, fm_feat_to], data_samples)
        losses.update(loss_decode)
        

        student_output = self.decode_head.forward([img_from, img_to, fm_feat_from, fm_feat_to])  # 学生模型的输出
        
        teacher_outputs = []
        
        self.teacher_s.eval()
        self.teacher_m.eval()
        self.teacher_l.eval()
        
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_s.extract_feat(img_from[i:i+1]), self.teacher_s.extract_feat(img_to[i:i+1])])
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_m.extract_feat(img_from[i:i+1]), self.teacher_m.extract_feat(img_to[i:i+1])])
                else:
                    teacher_output = self.teacher_l.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_l.extract_feat(img_from[i:i+1]), self.teacher_l.extract_feat(img_to[i:i+1])])

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses



    
@MODELS.register_module()
class SelfDistillBAN(BAN):
    def __init__(self,
                 distill_loss,
                 image_encoder: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 asymetric_input: bool = True,
                 encoder_resolution: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        super().__init__(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg)

        self.teacher_l = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_l)
        
        self.teacher_s = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_s)
        
        self.distill_loss = MODELS.build(distill_loss)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.image_encoder(fm_img_from)
        fm_feat_to = self.image_encoder(fm_img_to)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            [img_from, img_to, fm_feat_from, fm_feat_to], data_samples)
        losses.update(loss_decode)
        

        student_output = self.decode_head.forward([img_from, img_to, fm_feat_from, fm_feat_to])  # 学生模型的输出
        
        teacher_outputs = []
        teacher_l_inputs_from = []
        teacher_l_inputs_to = []
        teacher_s_inputs_from = []
        teacher_s_inputs_to = []
        teacher_l_data_samples = []
        teacher_s_data_samples = []
        
        # self.teacher_s.eval()
        # self.teacher_l.eval()
        
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.974:
                    teacher_output = self.teacher_s.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_s.extract_feat(img_from[i:i+1]), self.teacher_s.extract_feat(img_to[i:i+1])])
                    teacher_s_inputs_from.append(img_from[i:i+1])                    
                    teacher_s_inputs_to.append(img_to[i:i+1])
                    teacher_s_data_samples.append(data_sample)
                else:
                    teacher_output = self.teacher_l.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_l.extract_feat(img_from[i:i+1]), self.teacher_l.extract_feat(img_to[i:i+1])])
                    teacher_l_inputs_from.append(img_from[i:i+1])
                    teacher_l_inputs_to.append(img_to[i:i+1])
                    teacher_l_data_samples.append(data_sample)
                    
            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        # 对教师模型计算交叉熵损失
        if teacher_l_inputs_from:
            teacher_l_inputs_from = torch.cat(teacher_l_inputs_from, dim=0)
            teacher_l_inputs_to = torch.cat(teacher_l_inputs_to, dim=0)
            loss_teacher_l = self.teacher_l._decode_head_forward_train(
                [teacher_l_inputs_from, 
                 teacher_l_inputs_to, 
                 self.teacher_l.extract_feat(teacher_l_inputs_from), 
                 self.teacher_l.extract_feat(teacher_l_inputs_to), ], 
                teacher_l_data_samples
                )
            losses.update({f"teacher_l_{k}": v for k, v in loss_teacher_l.items()})

        if teacher_s_inputs_from:
            teacher_s_inputs_from = torch.cat(teacher_s_inputs_from, dim=0)
            teacher_s_inputs_to = torch.cat(teacher_s_inputs_to, dim=0)
            loss_teacher_s = self.teacher_s._decode_head_forward_train(
                [teacher_s_inputs_from, 
                 teacher_s_inputs_to, 
                 self.teacher_s.extract_feat(teacher_s_inputs_from), 
                 self.teacher_s.extract_feat(teacher_s_inputs_to), ], 
                teacher_s_data_samples
                )            
            losses.update({f"teacher_s_{k}": v for k, v in loss_teacher_s.items()})
            
        return losses