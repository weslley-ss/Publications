from pathlib import Path
import os
from collections.abc import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F

class DRIVE(Dataset):
    """Create a dataset object for holding a typical retina blood vessel dataset. 

    Note: __getitem__ returns a numpy array and not a pillow image because pillow
    does not support a negative ignore index (e.g. -100).
    """

    _HAS_TEST = True

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        channels: str = "all",
        keepdim: bool = False,
        return_mask: bool = False,
        ignore_index: int | None = None,
        normalize: bool = True,
        files: list | None = None,
        transforms: Callable | None = None,
    ):
        """
        root
            Root directory.
        split
            The split to use. Possible values are "train", "test" and "all"
        channels
            Image channels to use. Options are:
            "all": Use all channels
            "green": Use only the green channel
            "gray": Convert the image to grayscale
        keepdim
            If True, keeps the channel dimension in case `channels` is "green" or "gray"
        return_mask
            If True, also returns the retina mask
        ignore_index
            Index to put at the labels for pixels outside the mask (the retina). 
            If None, do nothing.
        normalize
            If True, divide the labels by 255 in case label.max()==255.
        files
            List of files to keep from the split. If None, use all files.
        transforms
            Transformations to apply to the images and the labels. If `return_mask` 
            is True, the transform needs to also accept the mask image as input.
        """
        
        self.root = Path(root)

        if split not in ["train", "test", "all"]:
            raise ValueError("Invalid split value. Must be 'train', 'test' or 'all'.")
        if channels not in ["all", "green", "gray"]:    
            raise ValueError("Invalid channels value. Must be 'all', 'green' or 'gray'.")
        
        if split=="test" and not self._HAS_TEST:
            raise ValueError("This dataset does not have a test split.")

        if split=="all":
            images, labels, masks = self._get_files(split="train")
            if self._HAS_TEST:
                images_t, labels_t, masks_t = self._get_files(split="test")
                images += images_t
                labels += labels_t
                masks += masks_t
        else:
            images, labels, masks = self._get_files(split=split)

        # Filter files if needed
        if files is not None:
            indices = search_files(files, images)
            images = [images[idx] for idx in indices]
            labels = [labels[idx] for idx in indices]
            masks = [masks[idx] for idx in indices]

        self.channels = channels
        self.keepdim = keepdim
        self.return_mask = return_mask
        self.ignore_index = ignore_index
        self.normalize = normalize

        self.images = sorted(images)
        self.labels = sorted(labels)
        self.masks = sorted(masks)
        self.classes = ["background", "vessel"]
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple:
        
        image = Image.open(self.images[idx])
        label = np.array(Image.open(self.labels[idx]), dtype=int)
        mask = np.array(Image.open(self.masks[idx]))

        # Select green channel or convert to gray
        if self.channels=="gray":
            image = image.convert("L")
        image = np.array(image)
        if self.channels=="green":
            image = image[:,:,1]
        if self.keepdim and image.ndim==2:
            image = np.expand_dims(image, axis=2)

        # Normalize label to [0,1] if in range [0,255]
        if self.normalize and label.max()==255:
            label = label//255
            mask = mask//255

        # Keep only first label channel if it is a color image
        if label.ndim==3:
            diff_pix = (label[:,:,0]!=label[:,:,1]).sum()
            diff_pix += (label[:,:,0]!=label[:,:,2]).sum()
            if diff_pix>0:
                raise ValueError("Label has multiple channels and they differ.")
            label = label[:,:,0]

        # Put ignore_index outside mask
        if self.ignore_index is not None:
            label[mask==0] = self.ignore_index

        output = image, label
        # Remember to also transform mask
        if self.return_mask:
            output += (mask,)
        if self.transforms is not None:
            output = self.transforms(*output)

        return output

    def __len__(self) -> int:
        return len(self.images)
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        if split=="train":
            root_split = self.root/"training"
            mask_str = "training"
        elif split=="test":
            root_split = self.root/"test"
            mask_str = "test"

        root_imgs = root_split/"images"
        root_labels = root_split/"1st_manual"
        root_masks = root_split/"mask"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            num, _ = file.split("_")
            images.append(root_imgs/file)
            labels.append(root_labels/f"{num}_manual1.gif")
            masks.append(root_masks/f"{num}_{mask_str}_mask.gif")

        return images, labels, masks

class ValidTransforms:
    """Validation transform that only resizes the image."""

    def __init__(self, resize_size = None, resize_target = True):
        self.resize_size = resize_size
        self.resize_target = resize_target

    def __call__(self, img, target):

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)
        
        if self.resize_size is not None:
            img = tv_transf_F.resize(img, self.resize_size)
            if self.resize_target:
                target = tv_transf_F.resize(target, 
                                            self.resize_size, 
                                            interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = img.float()/255
        target = target.to(dtype=torch.int64)[0]

        return img, target

def get_dataset_drive_test(
        dataset_path, 
        resize_size = None, 
        channels="all",
        ):
    """Get the DRIVE dataset used for training and test.

    Parameters
    ----------
    dataset_path
        Path to the dataset root folder
    split_strategy
        Strategy to split the dataset. Possible values are:
        "default": Use the default train/test split of the dataset
        "file": Use the train.csv and test.csv files to split the dataset
    resize_size
        Size to resize the images
    channels
        Image channels to use. Options are:
        "all": Use all channels
        "green": Use only the green channel
        "gray": Convert the image to grayscale
    use_ignore_index
        If True, the ignore index is set to 2. Otherwise, it is set to None
    """

    drive_params = {
        "channels":channels, "keepdim":True, "ignore_index":False
    }

    ds_train = DRIVE(dataset_path, split="train", **drive_params)
    ds_test = DRIVE(dataset_path, split="test", **drive_params)


    ds_train.transforms = ValidTransforms(resize_size)
    ds_test.transforms = ValidTransforms(resize_size)


    return ds_train, ds_test

def search_files(files: list[str], paths: list[Path]) -> list[int]:
    """Search for each file in list `paths` and return the indices of the files found
    in the paths list. Paths can then be fitlered as:

    >>> paths = [paths[idx] for idx in search_files(files, paths)] 

    Parameters
    ----------
    files
        List of file names to search   
    paths   
        List of paths.

    Returns
    -------
    List of indices of the files found in the paths list
    """

    indices = []
    for file in files:
        for idx, path in enumerate(paths):
            if file in str(path):
                indices.append(idx)

    return indices