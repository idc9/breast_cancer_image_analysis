import torch
import os
import numpy as np
import pandas as pd

from torchvision.transforms import Normalize, ToTensor, Compose

from cbcs_joint.patches.CBCSPatchGrid import CBCSPatchGrid
from cbcs_joint.patches.patch_features import compute_patch_features
from cbcs_joint.Paths import Paths
from cbcs_joint.cnn_models import load_cnn_model


os.makedirs(Paths().patches_dir, exist_ok=True)

#######################
# get patches dataset #
#######################
# compute the backgorund mask for each image, break into patches, throw out
# patches which have too much background

patch_kws = {'max_prop_background': .9,
             'patch_size': 200,
             'pad_image': 'div_200',
             'threshold_algo': 'triangle_otsu'}

patch_dataset = CBCSPatchGrid(**patch_kws)
patch_dataset.make_patch_grid()
patch_dataset.compute_pixel_stats(image_limit=10)
patch_dataset.save(os.path.join(Paths().patches_dir,
                                'patch_dataset'))

#############################
# Extract patch CNN features#
#############################

# CNN feature extraction model
model = load_cnn_model()

# patch image processing
# center and scale channels
channel_avg = patch_dataset.pixel_stats_['avg'] / 255
channel_std = np.sqrt(patch_dataset.pixel_stats_['var']) / 255
patch_transformer = Compose([ToTensor(),
                             Normalize(mean=channel_avg, std=channel_std)])

fpath = os.path.join(Paths().patches_dir, 'patch_features.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

compute_patch_features(patch_dataset=patch_dataset, model=model,
                       fpath=fpath,
                       patch_transformer=patch_transformer,
                       device=device)


###############################
# save core centroids to disk #
###############################

patch_feats = pd.read_csv(os.path.join(Paths().patches_dir,
                                       'patch_features.csv'),
                          index_col=['image', 'patch_idx'])
core_idxs = np.unique(patch_feats.index.get_level_values('image'))
core_centroids = pd.DataFrame(index=core_idxs, columns=patch_feats.columns)
