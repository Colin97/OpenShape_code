from argparse import ArgumentParser
import numpy as np
import torch
import os
import open_clip
from torch.utils.data import DataLoader, Dataset
from traceback import print_exc
import json

np.random.seed(42)
torch.manual_seed(42)

# pip install ftfy regex tqdm && pip install git+https://github.com/openai/CLIP.git && pip install open_clip_torch matplotlib

@torch.no_grad()
def extract_clip_feat(texts, clip_model,):
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    return clip_model.encode_text(text_tokens).float().cpu().numpy()

#load json file
lvis_anno = json.load(open('/datasets-slow1/Objaverse/lvis-annotations.json'))
names = sorted(list(lvis_anno.keys()))
print(names)
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir="/kaiming-fast-vol/workspace/open_clip_model/")
open_clip_model.cuda().eval()

results = []
for name in names:
    try:
        text_feat = extract_clip_feat([name], open_clip_model)
        results.append(text_feat)
        
    except Exception as e:
        print_exc()

results = np.concatenate(results)
print(results.shape)
np.save("/objaverse-processed/lvis_category_name_feat.npy", results, allow_pickle=True)
