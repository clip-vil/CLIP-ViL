from .clip_model.clip import load
from timm.models.vision_transformer import resize_pos_embed

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    ShapeSpec,
    get_norm,
)

__all__ = [
    "CLIP",
]

class CLIP(Backbone):
    """docstring for CLIP"""
    def __init__(self, model_type, dc5=False, frozen_stages=-1, out_features=None):
        super(CLIP, self).__init__()
        self.clip, _ = load(model_type, jit=False, dc5=dc5)
        for m in [self.clip.transformer, self.clip.token_embedding, self.clip.ln_final]:
            for param in m.parameters():
                param.requires_grad = False

        self.visual = self.clip.visual
        self.model_type = model_type
        if "ViT" in self.model_type:
            num_patches = 1000 * 1000 // 32 // 32
            print(self.visual.positional_embedding.shape, num_patches)
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768))
            pos_embed.weight = resize_pos_embed(self.visual.positional_embedding.unsqueeze(0), pos_embed.unsqueeze(0))

            self.visual.positional_embedding = pos_embed


        self._out_features = out_features
        self.dc5 = dc5
        self._out_feature_channels, self._out_feature_strides = {}, {}
        assert len(self._out_features)
        self._set_output_shape()

    def _set_output_shape(self):
        if "ViT" in self.model_type:
            self._out_feature_channels['res5'], self._out_feature_strides['res5'] = 768, 32
            # pass
        else:
            width = self.visual._inplanes // 8
            self._out_feature_channels['res2'], self._out_feature_strides['res2'] = width, 1
            self._out_feature_channels['res3'], self._out_feature_strides['res3'] = width*2, 2*2
            self._out_feature_channels['res4'], self._out_feature_strides['res4'] = width*4, 2*4
            # if self.dc5:
            #     self._out_feature_channels['res5'], self._out_feature_strides['res5'] = width*8, 1*8
            # else:
            self._out_feature_channels['res5'], self._out_feature_strides['res5'] = width*8, 2*8


    def _freeze_stages(self):
        if 'ViT' in self.model_type:
            if self.frozen_stages >= 0:
                for m in [self.visual]:
                    for param in m.parameters():
                        param.requires_grad = False
        else:
            if self.frozen_stages >= 0:
                for conv, bn in [(self.visual.conv1, self.visual.bn1), (self.visual.conv2, self.visual.bn2), 
                    (self.visual.conv3, self.visual.bn3)]:
                    bn.eval()
                    for m in [conv, bn]:
                        for param in m.parameters():
                            param.requires_grad = False                    

            for i in range(1, self.frozen_stages + 1):
                m = getattr(self.visual, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead! BGR"
        # BGR to RGB        
        outputs = {}
        x = x[:, [2, 1, 0], :, :]

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        if "ViT" in self.model_type:
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.visual.positional_embedding.to(x.dtype)[:x.shape[1], :]
            x = self.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            x = self.visual.ln_post(x[:, 1:, :])

            # if self.visual.proj is not None:
            #     x = x @ self.visual.proj

            # xs = []
            x = x.permute(0, 2, 1).reshape(B, x.shape[-1], H // 32, W // 32)

            outputs["res5"] = x
            # print(x.shape)
            # exit()
            # xs = [x.contiguous()]
        else:
            def stem(x):
                for conv, bn in [(self.visual.conv1, self.visual.bn1), (self.visual.conv2, self.visual.bn2), 
                    (self.visual.conv3, self.visual.bn3)]:
                    x = self.visual.relu(bn(conv(x)))
                x = self.visual.avgpool(x)
                return x

            xs = []
            # print(x.dtype, self.visual.conv1.weight.dtype)
            # x = x.type(self.visual.conv1.weight.dtype)
            x = stem(x)
            if "stem" in self._out_features:
                outputs["stem"] = x

            x = self.visual.layer1(x)
            if "res2" in self._out_features:
                outputs["res2"] = x

            x = self.visual.layer2(x)
            if "res3" in self._out_features:
                outputs["res3"] = x

            x = self.visual.layer3(x)
            if "res4" in self._out_features:
                outputs["res4"] = x

            x = self.visual.layer4(x)
            if "res5" in self._out_features:
                outputs["res5"] = x
            
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
        