import os
from joblib import load
import numpy as np

from cbcs_joint.Paths import Paths
from cbcs_joint.load_analysis_data import load_analysis_data
from cbcs_joint.cbcs_utils import get_cbcsid_group
from cbcs_joint.make_rpvs_for_component import viz_component
from cbcs_joint.utils import retain_pandas
from cbcs_joint.viz_utils import mpl_noaxis

# load pre-computed data e.g. patch features
data = load_analysis_data()
patch_dataset = data['patch_dataset']
patch_feats = data['patch_feats']
core_centroids = data['img_centroids']
subj_img_feats = data['subj_img_feats']
image_feats_processor = data['image_feats_processor']

avail_cbcsids = subj_img_feats.index
avail_cores = [i for i in core_centroids.index
               if get_cbcsid_group(i)[0] in avail_cbcsids]

# load precomputed AJIVE
ajive = load(os.path.join(Paths().results_dir, 'data', 'fit_ajive'))

# figure config
mpl_noaxis()

n_extreme_subjs = 15
n_patches_per_subj = 20
n_extreme_patches = 50

top_dir = Paths().results_dir


####################
# joint components #
####################

for comp in range(ajive.common.rank):

    comp_name = 'comp_{}'.format(comp + 1)

    subj_scores = ajive.common.scores(norm=True).iloc[:, comp]
    loading_vec = ajive.blocks['images'].common_loadings().iloc[:, comp]

    # transform patch features and project onto loadings vector
    patch_scores = retain_pandas(patch_feats,
                                 image_feats_processor.transform).\
        dot(loading_vec)

    # transform core features and project onto loadings vector
    core_scores = retain_pandas(core_centroids,
                                image_feats_processor.transform).\
        dot(loading_vec)

    viz_component(subj_scores=subj_scores,
                  core_scores=core_scores,
                  patch_scores=patch_scores,
                  patch_dataset=patch_dataset,
                  loading_vec=loading_vec,
                  comp_name=comp_name,
                  top_dir=top_dir,
                  signal_kind='common',
                  avail_cores=avail_cores,
                  n_extreme_subjs=n_extreme_subjs,
                  n_extreme_patches=n_extreme_patches,
                  n_patches_per_subj=n_patches_per_subj)


####################
# image individual #
####################

n_indiv_comps = 5

for comp in range(n_indiv_comps):
    comp_name = 'comp_{}'.format(comp + 1)
    print(comp_name)

    subj_scores = ajive.blocks['images'].\
        individual.scores(norm=True).iloc[:, comp]
    loading_vec = ajive.blocks['images'].\
        individual.loadings().iloc[:, comp]

    # transform patch features and project onto loadings vector
    patch_scores = \
        retain_pandas(patch_feats, image_feats_processor.transform).\
        dot(loading_vec)

    # transform core features and project onto loadings vector
    core_scores = retain_pandas(core_centroids,
                                image_feats_processor.transform).\
        dot(loading_vec)

    viz_component(subj_scores=subj_scores,
                  core_scores=core_scores,
                  patch_scores=patch_scores,
                  patch_dataset=patch_dataset,
                  loading_vec=loading_vec,
                  comp_name=comp_name,
                  top_dir=top_dir,
                  signal_kind='image_indiv',
                  avail_cores=avail_cores,
                  n_extreme_subjs=n_extreme_subjs,
                  n_extreme_patches=n_extreme_patches,
                  n_patches_per_subj=n_patches_per_subj)
