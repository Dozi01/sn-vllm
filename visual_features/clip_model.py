import __future__
import torch
import torchvision
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer, CLIPVisionModel
import numpy as np
import torchvision
from torch import dropout, nn
import gc


class CLIPNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        vision_tower_name = "openai/clip-vit-large-patch14"

        # Load vision tower and move to GPU
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name,
                                                    low_cpu_mem_usage=True)

        feat_dim = 1024

        self.inter = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

    def forward(self, video):

        out = self.vision_tower(video, output_hidden_states=True)

        # To create the features which we use as input
        select_hidden_state_layer = -2
        select_hidden_state = out.hidden_states[select_hidden_state_layer]
        batch_features = select_hidden_state[:, 1:]
        video_features = batch_features.detach().cpu()
        
        out = torch.mean(out.pooler_output, dim=0)
        out = out.unsqueeze(0)
        out = self.inter(out)
        out_off = self.fc_offence(out)
        out_act = self.fc_action(out)

        del out
        gc.collect()
        torch.cuda.empty_cache()

        return out_off.squeeze(), out_act.squeeze(), video_features

if __name__ == "__main__":
    Clip_model = CLIPNetwork().cuda()
    B = 16
    H = 224
    W = 224 
    C = 3
    x = torch.randn(B, C, H, W)
    x = x.cuda()
    print(f'input video shape : {x.shape}')
    y = Clip_model(x)
    print(f'offence prediction : {y[0].shape}')
    print(f'action prediction : {y[1].shape}')
    print(f'video features : {y[2].shape}')
