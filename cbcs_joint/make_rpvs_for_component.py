import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from imageio import imsave
from PIL import Image

from cbcs_joint.Paths import Paths
from cbcs_joint.viz_utils import savefig
from cbcs_joint.cbcs_utils import get_cbcsid_group
from cbcs_joint.patches.utils import plot_coord_ticks, get_patch_map
from cbcs_joint.jitter import jitter_hist


def viz_component(subj_scores, core_scores,
                  patch_scores, patch_dataset, avail_cores,
                  loading_vec, comp_name,
                  top_dir, signal_kind=None,
                  n_extreme_subjs=5, n_extreme_patches=20,
                  n_patches_per_subj=5):

    """
    Visualizations for interpreting a component. Looks at subject, core and
    patch level.
    """

    ###########################
    # histogram of all scores #
    ###########################
    scores_dir = get_and_make_dir(top_dir, signal_kind, 'scores')

    jitter_hist(subj_scores, hist_kws={'bins': 100})
    plt.xlabel('subject scores')
    savefig(os.path.join(scores_dir,
                         '{}_subject_scores.png'.format(comp_name)))

    jitter_hist(core_scores, hist_kws={'bins': 100})
    plt.xlabel('core scores')
    savefig(os.path.join(scores_dir,
                         '{}_core_scores.png'.format(comp_name)))

    jitter_hist(patch_scores, hist_kws={'bins': 100})
    plt.xlabel('patch scores')
    savefig(os.path.join(scores_dir,
                         '{}_patch_scores.png'.format(comp_name)))

    #################
    # subject level #
    #################

    # most extreme subjects by scores
    extreme_subjs = get_most_extreme(subj_scores, n_extreme_subjs)

    # extreme cores
    displayed_subj_scores = []
    for extr in extreme_subjs.keys():
        for subj_idx, cbcsid in enumerate(extreme_subjs[extr]):
            displayed_subj_scores.append(subj_scores[cbcsid])

            # cores for this subject
            subj_cores = [c for c in avail_cores if cbcsid in c]

            # move each core image to the directory
            for core_name in subj_cores:
                move_dir = get_and_make_dir(top_dir, signal_kind, 'cores', comp_name, extr)
                old_fpath = os.path.join(Paths().pro_image_dir, core_name)
                new_name = '{}_{}'.format(subj_idx, core_name)
                new_fpath = os.path.join(move_dir, new_name)

                copyfile(old_fpath, new_fpath)

    # plot subject level scores histogram
    jitter_hist(subj_scores.values, hist_kws={'bins': 100})
    plt.xlabel('{}, {} subject scores'.format(signal_kind, comp_name))
    for s in displayed_subj_scores:
        plt.axvline(s, color='red')
    save_dir = get_and_make_dir(top_dir, signal_kind, 'cores', comp_name)
    savefig(os.path.join(save_dir, 'scores.png'))

    ######################
    # patch for subjects #
    ######################
    # sorts patches by scores (ignoring subjects/core grouping)

    for extr in extreme_subjs.keys():
        folder_list = [signal_kind, 'core_patches', comp_name, extr]
        save_dir = get_and_make_dir(top_dir, *folder_list)

        for subj_rank, cbcsid in enumerate(extreme_subjs[extr]):

            # cores for this subject
            subj_cores = [c for c in avail_cores if cbcsid in c]

            # get top patches for this subject
            subj_patch_scores = patch_scores[[i for i in patch_scores.index
                                              if cbcsid in i[0]]]
            top_patch_idxs = list(get_most_extreme(subj_patch_scores,
                                                   n=n_patches_per_subj)[extr])

            plot_image_top_patches(cbcsid=cbcsid,
                                   subj_cores=subj_cores,
                                   top_patches=top_patch_idxs,
                                   patch_dataset=patch_dataset,
                                   patch_scores=patch_scores,
                                   n_patches_per_subj=n_patches_per_subj,
                                   inches=5)

            savefig(os.path.join(save_dir, '{}_{}.png'.format(subj_rank,
                                                              cbcsid)))

    ###############
    # patch level #
    ###############
    # shows the top patches and cores for the most extreme subjects

    # plot extreme patches
    extreme_patches = get_most_extreme(patch_scores, n_extreme_patches)

    # plot patches
    displayed_patch_scores = []
    for extr in extreme_patches.keys():
        folder_list = ['patches', comp_name, extr]
        save_dir = get_and_make_dir(top_dir, signal_kind, *folder_list)

        for patch_rank, patch_name in enumerate(extreme_patches[extr]):
            image_key, patch_idx = patch_name
            displayed_patch_scores.append(patch_scores[patch_name])
            image = patch_dataset.load_patches(image_key, patch_idx)
            name = '{}_patch_idx{}'.format(patch_rank, *patch_name)

            # save image
            imsave(os.path.join(save_dir, '{}.png'.format(name)), image)

    # plot patch level scores histogram
    jitter_hist(patch_scores.values, hist_kws={'bins': 100})
    plt.xlabel('{}, {} scores'.format(signal_kind, comp_name))
    for s in displayed_patch_scores:
        plt.axvline(s, color='red')
    save_dir = get_and_make_dir(top_dir, signal_kind, 'patches', comp_name)
    savefig(os.path.join(save_dir, 'scores.png'))


def plot_image_top_patches(cbcsid, subj_cores, top_patches,
                           patch_dataset,
                           patch_scores,
                           n_patches_per_subj=16, inches=5):

    # patches for each cores
    group2patch_idx = {get_cbcsid_group(c)[1]: [] for c in subj_cores}
    for r, i in enumerate(top_patches):
        group = get_cbcsid_group(i[0])[1]
        group2patch_idx[group].append(i[1])

    n_cols = max(4, len(subj_cores))
    n_rows = 2 + int(np.ceil(len(top_patches) / n_cols))
    plt.figure(figsize=[inches * n_cols, inches * n_rows])
    grid = plt.GridSpec(nrows=n_rows, ncols=n_cols)

    ##################
    # plot each core #
    ##################
    for i, core in enumerate(subj_cores):
        image = patch_dataset.load_image(core)

        plt.subplot(grid[0, i])
        plt.imshow(image)
        plt.title('{}, group {}'.format(*get_cbcsid_group(core)))

    #########################################
    # display patch map for each core cores #
    #########################################
    for i, core in enumerate(subj_cores):
        image = patch_dataset.load_image(core)
        _, group = get_cbcsid_group(core)

        # get patch map overlayed on image
        patch_map = get_patch_map(patch_dataset, image_key=core,
                                  patch_idxs=group2patch_idx[group])

        masked_image = np.array(overlay_alpha_mask(image, patch_map > 0.0))

        plt.subplot(grid[1, i])
        plt.imshow(masked_image)

    ####################
    # show top patches #
    ####################
    for rank, patch_name in enumerate(top_patches):

        image_key, patch_idx = patch_name
        cbcsid, group = get_cbcsid_group(image_key)

        patch = patch_dataset.load_patches(image_key=image_key,
                                           patch_index=patch_idx)
        top_left = patch_dataset.top_lefts_[image_key][patch_idx]

        r_idx = 2 + rank // n_cols
        c_idx = rank % n_cols

        plt.subplot(grid[r_idx, c_idx])
        plt.imshow(patch)
        plot_coord_ticks(top_left=top_left, size=patch_dataset.patch_size)
        plt.title('({}), {}, group {}, patch {} '.format(rank + 1, cbcsid, group, patch_idx))


def overlay_alpha_mask(image, mask, alpha=.2):
    """
    mask: Mask of pixels to highlight
    """

    alpha_mask = np.zeros(mask.shape)
    # alpha_mask[mask] = alpha_highlight
    alpha_mask[mask] = 1
    alpha_mask[~mask] = alpha
    alpha_mask = np.uint8(255 * alpha_mask)

    x = np.array(image)
    assert x.shape[2] == 3  # make sure image is rgb
    rbga = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1), dtype=np.uint8)
    rbga[:, :, 0:3] = x
    rbga[:, :, 3] = alpha_mask

    return Image.fromarray(rbga)


def get_most_extreme(values, n=5, ret='dict', middle=False):
    """
    Returns the most extreme positive and negative indices for a pd.Series
    """

    sorted_values = values.sort_values(ascending=False)

    most_pos = np.array(sorted_values.index[0:n])
    most_neg = np.array(sorted_values.index[-n:])[::-1]
    # most_neg = np.array(values.sort_values(ascending=True).index[0:n])

    if middle:
        middle = len(values) // 2
        left = middle - n // 2
        right = left + n
        middle_inds = np.array(sorted_values.index[left:right])

        if ret == 'dict':
            return {'pos': most_pos, 'middle': middle_inds, 'neg': most_neg}
        else:
            return most_pos, middle_inds, most_neg

    else:
        if ret == 'dict':
            return {'pos': most_pos, 'neg': most_neg}
        else:
            return most_pos, most_neg


def get_and_make_dir(*path):
    """
    Returns a path and creates the directory
    """

    paths_ = []
    for p in path:
        if p is not None:
            paths_.append(p)

    path = os.path.join(*paths_)
    os.makedirs(path, exist_ok=True)
    return path
