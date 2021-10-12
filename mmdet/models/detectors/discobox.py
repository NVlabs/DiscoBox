# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiscoBox. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from .single_stage_wsis import SingleStageWSInsDetector, SingleStageWSInsTSDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class DiscoBoxSOLOv2(SingleStageWSInsTSDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DiscoBoxSOLOv2, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                   test_cfg, pretrained)