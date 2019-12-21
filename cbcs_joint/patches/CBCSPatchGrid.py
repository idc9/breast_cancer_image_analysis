import numpy as np
from joblib import dump, load, Parallel, delayed
import os
from skimage.io import imread
from tqdm import tqdm

from cbcs_joint.Paths import Paths
from cbcs_joint.cbcs_utils import get_avail_images
from cbcs_joint.patches.utils import get_patch
from cbcs_joint.patches.utils import estimate_background, grid_coords_background_filtered, grid_gen, estimate_background_pixel, pad_image
from cbcs_joint.patches.stream import StreamAvg, StreamVar


class CBCSPatchGrid(object):
    """
    Breaks each core in the CBCS dataset into a grid of patches.

    Parameters
    ----------
    patch_size: int
        Dimensions of patches.

    pad_image: str ('div_X'), int, Nont
        Images are padded with their estimated background pixel to:
         be divisible by the patch size (e.g. 'div_200') or have a specified shape. If None or False, will not pad images.

    filter_background: bool
        Whether or not to ignore images with too much background.

    max_prop_background: float
        Maximum proportion background acceptable if we are filtering patches with too much background.

    threshold_algo: str
        Background estimation method, must be one of ['triangle', 'otsu', 'triange_otsu'].

    limit: None, int
        Maximum number of images to use. Useful for debugging purposes.

    """

    def __init__(self,
                 patch_size=200,
                 pad_image='div_200',
                 filter_background=True,
                 max_prop_background=0.9,
                 threshold_algo='triangle_otsu',
                 limit=None):

        self.max_prop_background = max_prop_background
        self.patch_size = patch_size
        self.pad_image = pad_image
        self.filter_background = filter_background
        self.threshold_algo = threshold_algo
        self.limit = limit

        img_fnames = get_avail_images('processed')
        if limit is not None:
            img_fnames = img_fnames[0:limit]

        self.top_lefts_ = {k: [] for k in img_fnames}

        self.image_shapes_ = {}
        self.pixel_stats_ = {'avg': None, 'var': None}
        self.image_shapes_ = {}
        self.background_thresholds_ = {}
        self.background_props_ = {}
        self.background_pixel_ = {}

    def save(self, fpath, compress=9):
        """
        Saves to disk, see documentation for joblib.dump
        """
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def image_keys(self):
        """
        Returns a list of image keys.
        """
        return list(self.top_lefts_.keys())

    def make_patch_grid(self, n_jobs=None):
        """
        Get the patch coordinates for every image in the dataset.

        Parameters
        ----------
        n_jobs: None, int
            Number of jobs to do in parallel.
        """

        image_keys = self.image_keys()

        if n_jobs is None or n_jobs == 1:
            coords = \
                [self._make_patch_coords_for_image(image_key=image_keys[i])
                 for i in range(len(image_keys))]

        else:
            coords = list(Parallel(n_jobs=n_jobs)
                          (delayed(self._make_patch_coords_for_image)
                           (image_key=image_keys[i])
                           for i in range(len(image_keys))))

        for i, image_key in enumerate(image_keys):
            self.top_lefts_[image_key].extend(coords[i])

        return self

    def _make_patch_coords_for_image(self, image_key):
        """
        Creates the patch grid for a single image.

        Output
        ------
        coords:

        """

        stride = self.patch_size

        if self.filter_background:
            image, background = self.load_image_and_background(image_key)
            coords = \
                grid_coords_background_filtered(size=stride,
                                                image_shape=image.shape,
                                                background=background,
                                                max_prop_background=self.max_prop_background,
                                                offset='center',
                                                extra_patch=False)

        else:
            image = self.load_image(image_key)

            coords = list(grid_gen(size=self.stride,
                                   image_shape=image.shape,
                                   offset='center',
                                   extra_patch=False))
        return coords

    def n_patches(self, image_key):
        return len(self.top_lefts_[image_key])

    def load_patches(self, image_key, patch_index=None):
        """
        Loads the patches for a given image.


        Parameters
        ----------
        image_key:
            Image key

        patch_index: int, None
            Index or indices of the patch to return. If None, returns all patches.

        Output
        ------
        Image patches.
        """

        image = self.load_image(image_key)
        patches = []
        if patch_index is None:
            array_output = True
            patch_index = range(self.n_patches(image_key))

        elif type(patch_index) in [list, np.array]:
            array_output = True

        else:
            array_output = False
            patch_index = [patch_index]

        for ind in patch_index:
            top_left = self.top_lefts_[image_key][ind]
            p = get_patch(image, top_left, size=self.patch_size)
            patches.append(p)

        if array_output:
            return patches
        else:
            return patches[0]

    def load_image(self, image_key):
        """
        Loads an image.

        Parameters
        ----------
        image_key: str
            The name of the image to load.

        """

        if self.pad_image is None:
            image = self._load_raw_image(image_key)
        else:
            # this will pad the image
            image, background = self.load_image_and_background(image_key)

        self.image_shapes_[image_key] = image.shape

        return image

    def load_image_and_background(self, image_key):

        # if no padding return the original image
        if self.pad_image is None:
            return self._load_raw_image_and_background(image_key)

        else:
            return self._load_padded_image_and_background(image_key)

    def _load_padded_image_and_background(self, image_key):

        image, background_mask = self._load_raw_image_and_background(image_key)

        if image_key not in self.background_pixel_.keys():
            pixel = estimate_background_pixel(image=image,
                                              mask=background_mask,
                                              method='median')

            self.background_pixel_[image_key] = pixel
        else:
            pixel = self.background_pixel_[image_key]

        # smallest number greater than current size, but divisible by div
        if 'div' in self.pad_image:
            div = int(self.pad_image.split('_')[1])
            current_size = image.shape[0:2]

            new_size = (int(np.ceil(current_size[0] / div) * div),
                        int(np.ceil(current_size[1] / div) * div))

        else:
            new_size = self.pad_image

        image = pad_image(image, new_size, pad_val=pixel)
        background_mask = pad_image(background_mask, new_size[0:2],
                                    pad_val=True)

        # update image shape
        self.image_shapes_[image_key] = image.shape

        return image, background_mask

    def _load_raw_image_and_background(self, image_key):
        """
        Loads an image an its estimated background mask.

        Output
        ------
        image, background_mask
        """

        image = self._load_raw_image(image_key)

        # if threshold is not already set then estimate it
        if image_key in self.background_thresholds_.keys():
            threshold = self.background_thresholds_[image_key]

            background_mask, _ = \
                estimate_background(image, threshold_algo=self.threshold_algo,
                                    threshold=threshold)

        else:
            background_mask, threshold = \
                estimate_background(image, threshold_algo=self.threshold_algo)

            self.background_thresholds_[image_key] = threshold

        # compute proportion background
        if image_key not in self.background_props_.keys():
            self.background_props_[image_key] = np.mean(background_mask)

        return image, background_mask

    def _load_raw_image(self, image_key):
        """
        Loads a core image.
        """
        fpath = os.path.join(Paths().pro_image_dir, image_key)
        image = imread(fpath)
        return image

    def patch_pixel_generator(self, image_limit=None, dtype=np.float64):
        for patch in self.patch_generator(image_limit=image_limit):
            patch = patch.reshape(-1, 3)
            for pixel in patch:
                if dtype is not None:
                    pixel = pixel.astype(np.float)
                yield pixel

    def patch_generator(self, image_limit=None):

        image_keys = self.image_keys()

        if image_limit is not None and image_limit < len(self.top_lefts_):
            image_keys = np.random.choice(image_keys, size=image_limit,
                                          replace=False)

        for image_key in image_keys:
            patches = self.load_patches(image_key)
            for p in patches:
                yield p

    def compute_pixel_stats(self, image_limit=None, dtype=np.float64):
        """
        Computes the channel-wise average and variance of each patch
        """
        channel_avgs = [StreamAvg(), StreamAvg(), StreamAvg()]
        channel_vars = [StreamVar(), StreamVar(), StreamVar()]

        pixel_stream = self.patch_pixel_generator(image_limit=image_limit,
                                                  dtype=np.float64)
        for pixel in tqdm(pixel_stream):
            for i, c in enumerate(pixel):
                channel_avgs[i].update(c)
                channel_vars[i].update(c)

        self.pixel_stats_['avg'] = np.array([channel_avgs[i].value()
                                             for i in range(3)])
        self.pixel_stats_['var'] = np.array([channel_vars[i].value()
                                             for i in range(3)])
