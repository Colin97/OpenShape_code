import numpy as np
import open3d as o3d
import random
import torch
import sys
from param import parse_args
import models
import MinkowskiEngine as ME
from utils.data import normalize_pc
from utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F

def load_ply(file_name, num_points=10000, y_up=True):
    pcd = o3d.io.read_point_cloud(file_name)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    n = xyz.shape[0]
    if n != num_points:
        idx = random.sample(range(n), num_points)
        xyz = xyz[idx]
        rgb = rgb[idx]
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz)
    if rgb is None:
        rgb = np.ones_like(rgb) * 0.4
    features = np.concatenate([xyz, rgb], axis=1)
    xyz = torch.from_numpy(xyz).type(torch.float32)
    features = torch.from_numpy(features).type(torch.float32)
    return ME.utils.batched_coordinates([xyz], dtype=torch.float32), features

def load_model(config, model_name="OpenShape/openshape-spconv-all"):
    model = models.make(config).cuda()

    if config.model.name.startswith('Mink'):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    return model

@torch.no_grad()
def extract_text_feat(texts, clip_model,):
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    return clip_model.encode_text(text_tokens)

@torch.no_grad()
def extract_image_feat(images, clip_model, clip_preprocess):
    image_tensors = [clip_preprocess(image) for image in images]
    image_tensors = torch.stack(image_tensors, dim=0).float().cuda()
    image_features = clip_model.encode_image(image_tensors)
    image_features = image_features.reshape((-1, image_features.shape[-1]))
    return image_features

print("loading OpenShape model...")
cli_args, extras = parse_args(sys.argv[1:])
config = load_config("src/configs/train.yaml", cli_args = vars(cli_args), extra_args = extras)
model = load_model(config)
model.eval()

print("loading OpenCLIP model...")
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir="/kaiming-fast-vol/workspace/open_clip_model/")
open_clip_model.cuda().eval()

print("extracting 3D shape feature...")
xyz, feat = load_ply("demo/pc.ply")
shape_feat = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 

print("extracting text features...")
texts = ["owl", "chicken", "penguin"]
text_feat = extract_text_feat(texts, open_clip_model)
print("texts: ", texts)
print("3D-text similarity: ", F.normalize(shape_feat, dim=1) @ F.normalize(text_feat, dim=1).T)

print("extracting image features...")
image_files = ["demo/a.jpg", "demo/b.jpg", "demo/c.jpg"]
images = [Image.open(f).convert("RGB") for f in image_files]
image_feat = extract_image_feat(images, open_clip_model, open_clip_preprocess)
print("image files: ", image_files)
print("3D-image similarity: ", F.normalize(shape_feat, dim=1) @ F.normalize(image_feat, dim=1).T)
