import torch
import codecs
import csv
from tqdm import tqdm

from torchvision.transforms import ToTensor


def compute_patch_features(patch_dataset, model, save='csv', fpath=None,
                           patch_transformer=ToTensor(),
                           device=None, limit=None):

    """
    Computes the features for each image patches.

    Parameters
    ----------

    patch_dataset: cbcs_joint.CBCSPatchGrid.CBCSPatchGrid

    model: pytorch module
        The feature extraction model


    fpath: str
        Where to save the features

    patch_transformer: callable, None
        Transformation to apply to the patch image before computing features.

    device: None, str
        Device for computations e.g. "cuda:0"
    """

    assert save in ['csv', 'return', 'dump']
    if save in ['csv', 'dump']:
        assert fpath is not None

    model = model.eval().float().to(device)

    if fpath[-4:] != '.csv':
        fpath += '.csv'
    fp = codecs.open(fpath, "w", 'utf-8')
    writer = csv.writer(fp)

    image_keys = patch_dataset.image_keys()
    patch_features = {}
    for i, image_key in tqdm(enumerate(image_keys)):

        if i == limit:
            break

        # load the patches for an image
        patches = patch_dataset.load_patches(image_key)

        patch_features[image_key] = []

        # compute features for each patch
        for patch_idx, patch in enumerate(patches):

            # image transformation + pytorch formatting
            patch = patch_transformer(patch)
            patch = patch.unsqueeze(0).float().to(device)

            # compute features
            with torch.no_grad():
                feats = model(patch)
                feats = feats.detach().cpu().squeeze().numpy()

            # flatten features for csv
            feats = list(feats.reshape(-1))

            # write column names
            if i == 0 and patch_idx == 0:
                header = ['feat_{}'.format(j) for j in range(len(feats))]
                header.insert(0, 'patch_idx')
                header.insert(0, 'image')
                writer.writerow(header)

            # write features to disk
            feats.insert(0, patch_idx)
            feats.insert(0, image_key)
            # feats.insert(0, '')
            writer.writerow(feats)

    fp.close()
