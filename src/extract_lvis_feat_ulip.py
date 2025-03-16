import json
import os
import random
import sys
from argparse import ArgumentParser
from itertools import chain
from traceback import print_exc

sys.path.append("/kaiming-fast-vol/workspace/ULIP_copy/")

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ULIP_models import ULIP_WITH_IMAGE
from utils.tokenizer import SimpleTokenizer


def generate_prompts(category):
    article = ["a", "the"][random.randint(0, 1)]
    if any([category.startswith(l) for l in "aeiou"]) and article == "a":
        article == "an"
    prompts = [
        f"There is {article} {category} in the scene",
        f"There is the {category} in the scene",
        f"a photo of {article} {category} in the scene",
        f"a photo of the {category} in the scene",
        f"a photo of one {category} in the scene",
        f"itap of {article} {category}",
        f"itap of my {category}",
        f"itap of the {category}",
        f"a photo of {article} {category}",
        f"a photo of my {category}",
        f"a photo of the {category}",
        f"a photo of one {category}",
        f"a photo of many {category}",
        f"a good photo of {article} {category}",
        f"a good photo of the {category}",
        f"a bad photo of {article} {category}",
        f"a bad photo of the {category}",
        f"a photo of a nice {category}",
        f"a photo of the nice {category}",
        f"a photo of a cool {category}",
        f"a photo of the cool {category}",
        f"a photo of a weird {category}",
        f"a photo of the weird {category}",
        f"a photo of a small {category}",
        f"a photo of the small {category}",
        f"a photo of a large {category}",
        f"a photo of the large {category}",
        f"a photo of a clean {category}",
        f"a photo of the clean {category}",
        f"a photo of a dirty {category}",
        f"a photo of the dirty {category}",
        f"a bright photo of {article} {category}",
        f"a bright photo of the {category}",
        f"a dark photo of {article} {category}",
        f"a dark photo of the {category}",
        f"a photo of a hard to see {category}",
        f"a photo of the hard to see {category}",
        f"a low resolution photo of {article} {category}",
        f"a low resolution photo of the {category}",
        f"a cropped photo of {article} {category}",
        f"a cropped photo of the {category}",
        f"a close-up photo of {article} {category}",
        f"a close-up photo of the {category}",
        f"a jpeg corrupted photo of {article} {category}",
        f"a jpeg corrupted photo of the {category}",
        f"a blurry photo of {article} {category}",
        f"a blurry photo of the {category}",
        f"a pixelated photo of {article} {category}",
        f"a pixelated photo of the {category}",
        f"a black and white photo of the {category}",
        f"a black and white photo of {article} {category}",
        f"a plastic {category}",
        f"the plastic {category}",
        f"a toy {category}",
        f"the toy {category}",
        f"a plushie {category}",
        f"the plushie {category}",
        f"a cartoon {category}",
        f"the cartoon {category}",
        f"an embroidered {category}",
        f"the embroidered {category}",
        f"a painting of the {category}",
        f"a painting of a {category}",
        f"a point cloud model of {category}"
    ]

    return prompts


def slip():
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    print("Vision model created.")

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = torch.nn.Identity()
    pc_feat_dims = 256
    print("PC model created.")
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims,
                            init_logit_scale=np.log(1/ 0.07), sep_head=False)
    print("Model ensembled.")

    # load the pretrained model
    pretrain_slip_model = torch.load('/kaiming-fast-vol/workspace/ULIP_copy/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
    pretrain_slip_model_params = pretrain_slip_model['state_dict']
    pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                    pretrain_slip_model_params.items()}

    for name, param in tqdm(model.named_parameters()):
        if name not in pretrain_slip_model_params:
            continue

        if isinstance(pretrain_slip_model_params[name], torch.nn.Parameter):
            param_new = pretrain_slip_model_params[name].data
        else:
            param_new = pretrain_slip_model_params[name]

        param.requires_grad = False
        print('load {} and freeze'.format(name))
        param.data.copy_(param_new)

    print("Pretrained weights loaded.")

    return model


def np_save(data, save_dir, save_path):
    os.makedirs(save_dir, exist_ok = True)
    np.save(save_path, data, allow_pickle = True)


class CLIPDataset(Dataset):

    def __init__(self, proc):
        self.data_list = []
        raw = json.load(open("/objaverse-processed/gpt4_filtering.json", "rb"))

        self.proc = proc
        keys = sorted(list(raw.keys()))
        self.data_list = [{"uid": k, "text": raw[k]["name"]}
            for i, k in enumerate(keys) if i % 16 == proc]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def extract_clip_feat(texts, clip_model, tokenizer):
    text_tokens = tokenizer(texts).cuda()
    text_feats = clip_model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.mean(dim=0)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.float().cpu().numpy()

    return text_feats


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model = slip()
    model.cuda().eval()
    tokenizer = SimpleTokenizer()

    lvis_anno = json.load(open('/datasets-slow1/Objaverse/lvis-annotations.json'))
    names = sorted(list(lvis_anno.keys()))

    results = []
    for name in tqdm(names):
        texts = generate_prompts(name)
        text_feat = extract_clip_feat(texts, model, tokenizer)
        results.append(text_feat)

    results = np.stack(results)
    print(results.shape)
    np.save("/objaverse-processed/lvis_category_name_feat_ulip.npy",
        results, allow_pickle=True)


if __name__ == "__main__":
    main()
