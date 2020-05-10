import glob
import os
import SimpleITK as sitk
import sys
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
        for mri in os.listdir(data_root):
            mri_folder = os.path.join(data_root, mri)

            files = glob.glob(mri_folder + "/*.nii")
            if len(files) > 0:
                for file in files:
                    print(file)
                    if file.split("/")[1] == "data.nii":
                        self.samples.append(sitk.ReadImage(files[0]))
                    elif file.split("/")[1] == "saliency.nii":
                        self.saliences.append(sitk.ReadImage(files[0]))
                    elif file.split("/")[1] == "segmentation.nii":
                        self.segmentations.append(sitk.ReadImage(files[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    if len(sys.argv) > 0:
        dataset = ImageNiftiDataset(str(sys.argv) + '/')
    else:
        dataset = ImageNiftiDataset('data/')
