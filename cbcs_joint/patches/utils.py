import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.color import rgb2grey
import numbers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from math import ceil, floor
from skimage.util import pad

from cbcs_joint.cbcs_utils import get_cbcsid_group


def get_patch(image, top_left, size):
    """
    Extracts a patch from an image.


    Parameters
    ----------

    image: shape (height, width, n_channels)
        Image to extract the patch from.


    top_left: tuple (h, w)
        Coordinates of the top left pixel of the patch. The first
        coordinate is how far down and the second coordinate is how
        far to the right.

    size: int, tuple
        Size of the patch to extract.

    Output
    ------
    array-like, shape (size[0], size[1], n_channels)

    """
    if type(size) in [float, int]:
        size = (int(size), int(size))

    if top_left[0] + size[0] > image.shape[0]:
        raise ValueError('Patch goes off image: top_left[0] + size[0] >'
                         'image.shape[0] ({} + {} > {})'.
                         format(top_left[0], size[0], image.shape[0]))

    if top_left[1] + size[1] > image.shape[1]:
        raise ValueError('Patch goes off image: top_left[1] + size[1] >'
                         'image.shape[1] ({} + {} > {})'.
                         format(top_left[1], size[1], image.shape[1]))

    return image[top_left[0]:(top_left[0] + size[0]),
                 top_left[1]:(top_left[1] + size[1]), ...]


def rand_patch_coords(size, image_shape, n_patches=None, rng=None,
                      n_jobs=None):

    kwargs = {'size': size, 'image_shape': image_shape, 'rng': rng}

    if n_patches is None:
        return rand_patch_coords_(**kwargs)

    return parallel_sample(fun=rand_patch_coords_,
                           n_jobs=n_jobs, n_samples=n_patches,
                           **kwargs)


def rand_patch_coords_(size, image_shape, rng=None):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    if rng is not None:
        if type(rng) == int:
            rng = np.random.RandomState(rng)

        top_left = (rng.randint(low=0, high=image_shape[0] - size[0]),
                    rng.randint(low=0, high=image_shape[1] - size[1]))

    else:
        top_left = (np.random.randint(low=0, high=image_shape[0] - size[0]),
                    np.random.randint(low=0, high=image_shape[1] - size[1]))

    return top_left


def estimate_background(image, threshold_algo='triangle_otsu',
                        threshold=None):
    """
    Estimates the background and pixel intesnity background theshold for an image. Note intensity is calculated using skimage.color.rgb2grey

    from import threshold_triangle, threshold_otsu

    Parameters
    ----------
    image: array-like, (height, width, n_channels)
        Image whose background to estimate.

    threshold_algo: str, ['otsu', 'triangle', 'triangle_otsu']
        Thresholding algorithm to estimate the background.
        'otsu': skimage.filters.threshold_otsu
        'triangle': skimage.filters.threshold_triangle
        'triangle_otsu': .9 * triangle + .1 * otsu


    threshold: None, float, int
        User provided threshold. If None, will be estimated using one
        of the thresholding algorithms.


    Output
    ------
    background_mask, threshold

    background_mask: array-like, (height, width)
        The True/False mask of estimated background pixels.

    threshold: float
        The (lower bound) backgound threshold intensity for the image.

    """

    grayscale_image = rgb2grey(image)

    if threshold is None:
        if threshold_algo == 'otsu':
            threshold = threshold_otsu(grayscale_image)

        elif threshold_algo == 'triangle':
            threshold = threshold_triangle(grayscale_image)

        elif threshold_algo == 'triangle_otsu':
            triangle = threshold_triangle(grayscale_image)
            otsu = threshold_otsu(grayscale_image)

            threshold = .9 * triangle + .1 * otsu

        else:
            raise ValueError('threshold_algo = {} is invalid argument'.format(threshold_algo))

    background_mask = grayscale_image > threshold

    return background_mask, threshold


# class PatchSamplingError(Exception):
#     """Sampling error"""
#     pass


def rand_coords_background_filt(size, image_shape,
                                background, max_prop_background,
                                n_patches=None,
                                max_tries=100, rng=None, n_jobs=None):

    kwargs = {'size': size,
              'image_shape': image_shape,
              'background': background,
              'max_prop_background': max_prop_background,
              'max_tries': max_tries,
              'rng': rng}

    if n_patches is None:
        return rand_coords_background_filt_(**kwargs)

    return parallel_sample(fun=rand_coords_background_filt_,
                           n_jobs=n_jobs, n_samples=n_patches,
                           **kwargs)


def rand_coords_background_filt_(size, image_shape,
                                 background, max_prop_background,
                                 max_tries=100, rng=None):
    """
    Samples patch coordinates uniformly at random. Filters patches
    which contain too much background.

    Parameters
    ----------
    """
    for t in range(max_tries):
        tl = rand_patch_coords(size=size, image_shape=image_shape, rng=rng)

        patch_background = get_patch(background, tl, size=size)

        if np.mean(patch_background) <= max_prop_background:
            return tl

        # if we exhausted all the tries give up
        if t == max_tries - 1:
            return None
            # raise PatchSamplingError('ran out of tries')


def grid_gen(size, image_shape,
             offset='center', extra_patch=False):
    """

    Generates the coordinates for a grid of patches going from left
    to right then top to bottom.

    Parameters
    ----------
    size: int or (int, int)
        Size of the patches (height, width).

    image_shape: int or (int, int)
        Size of the image

    offset: (int, int)

    extra_patch: bool
        Add extra patch at the end so every pixel is included.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    if isinstance(image_shape, numbers.Number):
        image_shape = (int(image_shape), int(image_shape))

    if isinstance(offset, numbers.Number):
        offset = (int(offset), int(offset))

    if offset == 'center':
        offset = ((image_shape[0] % size[0]) // 2,
                  (image_shape[1] % size[1]) // 2)

    elif offset is None:
        offset = (0, 0)

    else:
        assert len(offset) == 2

    n_rows = (image_shape[0] - offset[0]) // size[0]
    n_cols = (image_shape[1] - offset[1]) // size[1]

    if extra_patch:
        n_cols += 1
        n_rows += 1

    for r in range(n_rows):
        for c in range(n_cols):
            coord = (r * size[0] + offset[0], c * size[1] + offset[1])

            if extra_patch:
                if r == n_cols - 1:
                    x = coord[0] - ((image_shape[0] - offset[0]) % size[0])
                else:
                    x = coord[0]

                if c == n_rows - 1:
                    y = coord[1] - ((image_shape[1] - offset[1]) % size[1])
                else:
                    y = coord[1]

                coord = (x, y)

            yield coord


def grid_coords_background_filtered(size, image_shape,
                                    background, max_prop_background,
                                    offset='center', extra_patch=False):

    top_lefts = []
    for tl in grid_gen(size=size, image_shape=image_shape,
                       offset=offset, extra_patch=extra_patch):
        patch_background = get_patch(background, top_left=tl, size=size)

        if np.mean(patch_background) <= max_prop_background:
            top_lefts.append(tl)

    return top_lefts


def parallel_sample(fun, n_samples, n_jobs=None, **kwargs):

    if n_jobs is None:
        return [fun(**kwargs) for _ in range(n_samples)]

    return list(Parallel(n_jobs=n_jobs)(delayed(fun)(**kwargs)
                                        for _ in range(n_samples)))


def get_patch_map(patch_dataset, image_key, patch_idxs=None):
    """
    Returns the patch map for a given image

    Parameters
    ----------
    image_key:
        Which image.

    mask: bool
        If true, returns a True/False array. Otherwise, counts the
        number of times each pixel shows up in a patch.
    """
    # TODO: deprecate when new patch_dataset comes online
    if patch_idxs is None:
        patch_idxs = range(len(patch_dataset.top_lefts_[image_key]))

    image_shape = patch_dataset.load_image(image_key).shape[0:2]
    patch_map = np.zeros(image_shape)
    size = (patch_dataset.patch_size, patch_dataset.patch_size)
    for idx in patch_idxs:
        top_left = patch_dataset.top_lefts_[image_key][idx]
        patch_map[top_left[0]:(top_left[0] + size[1]),
                  top_left[1]:(top_left[1] + size[1])] += 1.0

    return patch_map


def plot_coord_ticks(top_left, size, tick_spacing=50, n_ticks=5):
    if isinstance(size, numbers.Number):
        size = (size, size)

    xmin = top_left[1]
    ymin = top_left[0]

    toc_locs = np.arange(n_ticks) * tick_spacing

    plt.xticks(ticks=toc_locs, labels=toc_locs + xmin)
    plt.yticks(ticks=toc_locs, labels=ymin + toc_locs)


def estimate_background_pixel(image, mask, method='median'):
    """
    Estimates the typical background pixels of an image.

    Parameters
    ----------
    image: array-like (height, width, n_channels)
        The image.

    mask: array-like (height, width)
        True/False array for background pixels (True means a pixel is background.)

    method: str, ['mean', 'median']
        Use the channel wise mean or median to estimate typical pixel.
    """

    background_pixels = image.reshape(-1, 3)[mask.reshape(-1), :]

    if method == 'mean':
        est_background_pixel = np.mean(background_pixels, 0)
    elif method == 'median':
        est_background_pixel = np.median(background_pixels, 0)
    else:
        raise ValueError('method should be either mean or median, not {}'.format(method))

    return est_background_pixel


def get_subj_background(patch_dataset, avail_cbcsids=None):
    """
    Returns the average proportion background by subject
    """

    # extract proportion background
    background_props = pd.Series(patch_dataset.background_props_)
    cbcsids = [get_cbcsid_group(i)[0] for i in background_props.index]
    df = pd.DataFrame()
    df['background'] = background_props
    df['cbcsid'] = cbcsids
    df = df.groupby('cbcsid').mean()

    if avail_cbcsids is not None:
        df = df.loc[avail_cbcsids]

    return df['background']


def get_subj_background_intensity(patch_dataset, avail_cbcsids=None):

    def intensity(pixel, weighted=True):
        pixel = np.array(pixel)

        if weighted:
            c = np.array([0.2125, 0.7154, 0.0721])
        else:
            c = np.array([1 / 3, 1 / 3, 1 / 3])

        return np.dot(c, pixel)

    # compute intensity of background pixel
    intensities = {}
    for k, pixel in patch_dataset.background_pixel_.items():
        intensities[k] = intensity(pixel, weighted=True)

    intensities = pd.Series(intensities)
    cbcsids = [get_cbcsid_group(i)[0] for i in intensities.index]
    df = pd.DataFrame()
    df['background_intensity'] = intensities
    df['cbcsid'] = cbcsids
    df = df.groupby('cbcsid').mean()

    if avail_cbcsids is not None:
        df = df.loc[avail_cbcsids]

    return df['background_intensity']


def pad_image(image, new_size, pad_val):
    """
    Pads an image to a desired size.

    Parameters
    ----------
    image (ndarray): (height, width, n_channels)
        Image to pad.

    new_size: int, tuple, (new_heght, new_width)
        Image will be padded to (new_height, new_width, n_channels)

    pad_vad: float, listlike value to pad with

    """
    if isinstance(new_size, numbers.Number):
        new_size = (new_size, new_size)

    if image.shape[0] > new_size[0]:
        print('WARNING: image height larger than desired cnn image size')
        return image

    if image.shape[1] > new_size[1]:
        print('WARNING: image width larger than desiered cnn image size')
        return image

    # width padding
    width_diff = new_size[1] - image.shape[1]

    # how much padding to add
    left = floor(width_diff / 2)
    right = ceil(width_diff / 2)

    height_diff = new_size[0] - image.shape[0]

    # how much padding to add
    top = floor(height_diff / 2)
    bottom = ceil(height_diff / 2)

    pad_width = ((top, bottom), (left, right), (0, 0))
    # make work with 2-d arrays
    if len(image.shape) == 2:
        pad_width = pad_width[0:2]

    if isinstance(pad_val, numbers.Number):
        return pad(image, pad_width, mode='constant', constant_values=pad_val)

    else:
        n_channels = image.shape[2]
        return np.stack([pad(image[:, :, c], pad_width[0:2], mode='constant',
                             constant_values=pad_val[c])
                         for c in range(n_channels)], axis=2)
