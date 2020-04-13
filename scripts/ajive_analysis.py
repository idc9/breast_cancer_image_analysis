import os
from joblib import dump
import matplotlib.pyplot as plt

from jive.AJIVE import AJIVE

from explore.BlockBlock import BlockBlock
from explore.Base import Union

from cbcs_joint.load_analysis_data import load_analysis_data
from cbcs_joint.viz_utils import savefig, mpl_noaxis
from cbcs_joint.Paths import Paths

# make directories for saved results
os.makedirs(os.path.join(Paths().results_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'common',
                         'loadings'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'genetic_indiv',
                         'loadings'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'image_indiv'), exist_ok=True)


# load pre-computed data e.g. patch features
data = load_analysis_data(load_patch_feats=False)
subj_img_feats = data['subj_img_feats']
genes = data['genes']
clinical_data = data['clinical_data']

# initial signal ranks determined from PCA scree plots
init_signal_ranks = {'images': 81, 'genes': 30}

# run AJIVE
ajive = AJIVE(init_signal_ranks=init_signal_ranks,
              n_wedin_samples=1000, n_randdir_samples=1000,
              zero_index_names=False, n_jobs=-1, store_full=False)
ajive = ajive.fit({'images': subj_img_feats, 'genes': genes})

dump(ajive, os.path.join(Paths().results_dir, 'data', 'fit_ajive'))

#####################
# AJIVE diagnostics #
#####################

# diagnostic plot
plt.figure(figsize=[10, 10])
ajive.plot_joint_diagnostic()
savefig(os.path.join(Paths().results_dir, 'ajive_diagnostic.png'))

#######################
# plot PAM50 loadings #
#######################

# set visualization configs
mpl_noaxis(labels=True)

n_genes = 50
inches = 5
height_scale = n_genes // 25
load_figsize = (inches, height_scale * inches)

# common loadings
load_dir = os.path.join(Paths().results_dir, 'common', 'loadings')
os.makedirs(load_dir, exist_ok=True)
for r in range(ajive.common.rank):
    plt.figure(figsize=load_figsize)
    ajive.blocks['genes'].plot_common_loading(r)
    plt.title('common component {}'.format(r + 1))
    savefig(os.path.join(load_dir, 'loadings_comp_{}.png'.format(r + 1)))


# genetic individual loadings
load_dir = os.path.join(Paths().results_dir, 'genetic_indiv', 'loadings')
os.makedirs(load_dir, exist_ok=True)
n_indiv_comps = min(5, ajive.blocks['genes'].individual.rank)
for r in range(n_indiv_comps):
    plt.figure(figsize=load_figsize)
    ajive.blocks['genes'].individual.plot_loading(r)
    plt.title('genetic individual component {}'.format(r + 1))
    savefig(os.path.join(load_dir, 'loadings_comp_{}.png'.format(r + 1)))

#########################################
# compare AJIVE scores to clinical data #
#########################################
# see documentation of explore package
# BlockBlock compares all variables from one block (AJIVE scores) to
# all variables of another block (clinical variables)
# and adjusts for multiple testing

comparision_kws = {'alpha': 0.05,
                   'multi_test': 'fdr_bh',
                   'cat_test': 'auc',  # equivalent to a Mann-Whitney test
                   'multi_cat': 'ovo',
                   'nan_how': 'drop'}


common_scd = BlockBlock(**comparision_kws)
common_scd.fit(ajive.common.scores(norm=True),
               clinical_data)

gene_indiv_scd = BlockBlock(**comparision_kws)
gene_indiv_scd = gene_indiv_scd.\
    fit(ajive.blocks['genes'].individual.scores_.iloc[:, 0:5], clinical_data)

image_indiv_scd = BlockBlock(**comparision_kws)
image_indiv_scd = BlockBlock().\
    fit(ajive.blocks['images'].individual.scores_.iloc[:, 0:5], clinical_data)

all_tests = Union().add_tests([('common', common_scd),
                               ('gene_indiv', gene_indiv_scd),
                               ('image_indiv', image_indiv_scd)])

all_tests.correct_multi_tests()

dump(all_tests, os.path.join(Paths().results_dir, 'data',
                             'clinical_data_comparisions'))

inches = 6

# common
n_row, n_col = common_scd.comparisons_.shape
plt.figure(figsize=(inches * n_col, inches * n_row))
common_scd.plot()
savefig(os.path.join(Paths().results_dir, 'common',
                     'cns_vs_clinical_data.png'), dpi=100)

# genetic individual
n_row, n_col = gene_indiv_scd.comparisons_.shape
plt.figure(figsize=(inches * n_col, inches * n_row))
gene_indiv_scd.plot()
savefig(os.path.join(Paths().results_dir, 'genetic_indiv',
                     'genetic_indiv_vs_clinical_data.png'), dpi=100)

# image individual
n_row, n_col = image_indiv_scd.comparisons_.shape
plt.figure(figsize=(inches * n_col, inches * n_row))
image_indiv_scd.plot()
savefig(os.path.join(Paths().results_dir, 'image_indiv',
                     'image_indiv_vs_clinical_data.png'), dpi=100)
