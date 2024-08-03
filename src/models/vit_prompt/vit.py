#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from src.models.vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from src.utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            print(prompt_dim)
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            print('patch_size', patch_size, reduce(mul, patch_size, 1), prompt_dim)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            print(val)  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))

            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"] - 1
                print(total_d_layer)
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        self.tune = nn.Linear(3 * 224 * 224, num_tokens * prompt_dim)

    def IIC(self, x):
        print(f'IIC.shape:{x.shape}')
        # x should be in range 0-1
        # print(x.shape)#[197,768]
        b_channel, g_channel, r_channel = x.unbind(dim=0)

        # Compute chromaticity
        total = r_channel + g_channel + b_channel
        c_b = torch.where(total == 0, torch.zeros_like(total), b_channel / total)
        c_g = torch.where(total == 0, torch.zeros_like(total), g_channel / total)
        c_r = torch.where(total == 0, torch.zeros_like(total), r_channel / total)

        # Normalize chromaticity
        c_mean = torch.mean(torch.stack([c_b, c_g, c_r]), dim=0)
        c_std = torch.std(torch.stack([c_b, c_g, c_r]), dim=0)
        # c_b_norm = (c_b - c_mean) / c_std * 64 + 127
        # c_g_norm = (c_g - c_mean) / c_std * 64 + 127
        # c_r_norm = (c_r - c_mean) / c_std * 64 + 127
        c_b_norm = (c_b - c_mean) / c_std
        c_g_norm = (c_g - c_mean) / c_std
        c_r_norm = (c_r - c_mean) / c_std

        # Combine the three channels back into an image
        c_img = torch.stack((c_b_norm, c_g_norm, c_r_norm), dim=1)
        # print(c_img.shape)

        return c_img

    def IIC_batch(self, batch_x):
        batch_y = []

        for x in batch_x:
            c_img = self.IIC(x)
            batch_y.append(c_img)

        batch_y_tensor = torch.stack(batch_y, dim=0)
        return batch_y_tensor


    def incorporate_prompt(self, x, y):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        # y = self.IIC_batch(x)

        y_flattened = y.view(B, -1)

        tuned_x = self.tune(y_flattened)

        num_tokens = 100
        prompt_dim = 768
        tuned_x = tuned_x.view(B, -1, prompt_dim)

        x_combined = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        return x_combined


    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]
        # print("num_layers",num_layers)
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1 + self.num_tokens):, :]
                    ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # print(x.shape)  # [B, 3, 224, 224]
        y = self.IIC_batch(x)
        # this is the default version:
        embedding_output = self.incorporate_prompt(x, y)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
            self, prompt_cfg, model_type,
            img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights


from src.configs.config_node import CfgNode

PROMPT = CfgNode()
PROMPT.NUM_TOKENS = 100
PROMPT.LOCATION = "prepend"
PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
PROMPT.CLSEMB_FOLDER = ""
PROMPT.CLSEMB_PATH = ""
PROMPT.PROJECT = -1  # "projection mlp hidden dim"
PROMPT.DEEP = True  # "whether do deep prompt or not, only for prepend location"
PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt

PROMPT.VIT_POOL_TYPE = "original"
PROMPT.DROPOUT = 0.1
PROMPT.SAVE_FOR_EACH_EPOCH = False

# from src.configs import vit_configs as configs
#
# model = PromptedTransformer(PROMPT,configs.get_b16_config(), 224,vis=False)
#
# out = model(torch.rand(1,3,224,224))
#
# print(model)
# print(model)
