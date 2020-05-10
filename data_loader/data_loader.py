import glob
import os
import SimpleITK as sitk
from torch.utils.data import Dataset


class ImageNiftiDataset(Dataset):
    """.nii images dataset"""

    def __init__(self, data_root):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.data_root = data_root
        self.samples = []
        self.saliences = []
        self.segmentations = []

        # all data in folder
        for mri_dir in os.scandir(data_root):
            if mri_dir.is_dir():
                self._load_data_folder(mri_dir)

    def _load_data_folder(self, mri_dir):
        for f in glob.glob(mri_dir.path + "/*.nii"):
            name = f.split("/")[-1]
            if name == "data.nii":
                self.samples.append(sitk.ReadImage(f))
            elif name == "saliency.nii":
                self.saliences.append(sitk.ReadImage(f))
            elif name == "segmentation.nii":
                self.segmentations.append(sitk.ReadImage(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
