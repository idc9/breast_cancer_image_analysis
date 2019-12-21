import os


class Paths(object):
    """
    Contains paths to directories used in the analysis.

    The user should modify data_dir; everything else should work from there.
    """
    def __init__(self):

        # top level data directory for the analysis
        # The user should modify this attribute before installing the package!
        self.data_dir = '/Users/iaincarmichael/data/breast_cancer_image_analysis/repro_aoas/'

        # the following files may be requested from the CBCS steering committee
        # should be placed in the data directory
        # cbcs3_ge_020719.csv (the gene expression data)
        # cleaned_clinical_data.csv (the clinical variables)
        # pam50_genes.txt (list of the 50 PAM50 genes)

        # patch features and coordinates
        self.patches_dir = os.path.join(self.data_dir, 'patches')

        # raw images are saved here and with file names in the form of CBCS3_HE_<CBCS_ID>_group_<CORE_ID>_image.jpg
        # it is ok to not include the raw images
        self.raw_image_dir = os.path.join(self.data_dir, 'raw_images')

        # processed images are saved here with file names in the form of
        # CBCS3_HE_<CBCS_ID>_group_<CORE_ID>_image_restained.png
        # for the CBCS paper we applied a restaining procedure
        # the restained images may be requested from
        #  the CBCS steering committee
        self.pro_image_dir = os.path.join(self.data_dir, 'processed_images')

        # where to save results of analyses
        self.results_dir = os.path.join(self.data_dir, 'results')

    def make_directories(self):
        """
        Creates the top level data directories.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.patches_dir, exist_ok=True)
        os.makedirs(self.raw_image_dir, exist_ok=True)
        os.makedirs(self.pro_image_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
