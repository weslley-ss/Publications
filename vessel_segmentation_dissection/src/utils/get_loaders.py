import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage import measure
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F
from torchvision import tv_tensors
from scipy import ndimage
from skimage.morphology import skeletonize

def erosion(img, target):
    img = np.array(img)
    target = np.array(target)

    # Realizando a erosão pelo scipy
    erosion = ndimage.binary_erosion(target)*1

    # Subtraindo a erosão da imagem original
    subtraction_result = target - erosion

    # Nova imagem só com os pontos do contorno
    img_result = np.zeros_like(img)
    img_result[subtraction_result == 1] = img[subtraction_result == 1]
    
    return Image.fromarray(img_result)

def skeletization(img, target):
    img = np.array(img)
    target = np.array(target)

    # Fazendo a esqueletização
    skeleton = skeletonize(target)
    
    # Nova imagem só com o esqueleto dos vasos
    img_result = np.zeros_like(img)
    img_result[skeleton == True] = img[skeleton == True]

    return Image.fromarray(img_result)

def contour_ratio (img, target, variation):
    img = np.array(img)
    target = np.array(target)

    
    # contour (by erosion)
    erosion = ndimage.binary_erosion(target).astype(int)
    img_cont = target - erosion

    # skeleton
    img_skel = skeletonize(target)

    # dilation of skeleton 
    struct = ndimage.generate_binary_structure(2, 1)
    img_skel_dil = ndimage.binary_dilation(
        img_skel, structure=struct, iterations=variation
    ).astype(int)

    # remove central region 
    img_rem = target - img_skel_dil
    img_rem[img_rem < 0] = 0

    # keep contour on image
    img_rem[img_cont > 0] = 1

    # aply mask to image
    img_result = np.zeros_like(img)
    img_result[img_rem == 1] = img[img_rem == 1]

    return Image.fromarray(img_result)
    
class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None, channels='all'):
        df = pd.read_csv(csv_path)
        self.root = osp.dirname(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.channels = channels
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = measure.regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(osp.join(self.root,self.im_list.iloc[index]))
        target = Image.open(osp.join(self.root,self.gt_list[index]))
        mask = Image.open(osp.join(self.root,self.mask_list[index])).convert('L')

        if self.channels=='gray':
            img = img.convert('L')
        elif self.channels=='green':
            img = Image.fromarray(np.array(img)[:,:,1])

        img, target, mask = self.crop_to_fov(img, target, mask)

        target = np.array(self.label_encoding(target))

        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # QUICK HACK FOR PSEUDO_SEG IN VESSELS, BUT IT SPOILS A/V
        if len(self.label_values)==2: # vessel segmentation case
            target = target.float()
            if torch.max(target)>1:
                target= target.float()/255

        return img, target

    def __len__(self):
        return len(self.im_list)

class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size, channels='all'):
        df = pd.read_csv(csv_path)
        self.root = osp.dirname(csv_path)
        self.im_list = df.im_paths
        self.mask_list = df.mask_paths
        self.channels = channels
        self.tranforms = TestTransforms(tg_size)

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = measure.regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]
    
    def crop(self, img, bbox):
        minr, minc, maxr, maxc = bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop

    def __getitem__(self, index):
        # # load image and mask
        img = Image.open(osp.join(self.root,self.im_list[index]))
        mask = Image.open(osp.join(self.root,self.mask_list[index])).convert('L')

        if self.channels=='gray':
            img = img.convert('L')
        elif self.channels=='green':
            img = Image.fromarray(np.array(img)[:,:,1])

        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        img = self.tranforms(img)

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

class TrainTransforms:

    def __init__(self, tg_size, transform, variation):

        self.tg_size = tg_size
        self.transform_method = transform
        self.variation = int(variation.split('_')[-1])  # get only the number part

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = tv_transf.RandomChoice((scale, transl, rotate))

        #brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        #jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)

        hflip = tv_transf.RandomHorizontalFlip()
        vflip = tv_transf.RandomVerticalFlip()

        to_dtype = tv_transf.ToDtype(
            {
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64
            },
            scale=True   # Mask is not scaled
        )

        unwrap = tv_transf.ToPureTensor()

        self.transform = tv_transf.Compose((
            scale_transl_rot,
            #jitter,
            hflip,
            vflip,
            to_dtype,
            unwrap
        ))

    def __call__(self, img, target):
        img = tv_transf_F.resize(img, self.tg_size)
        # NEAREST_EXACT has a 0.01 better Dice score than NEAREST. The
        # object oriented version of resize uses NEAREST, thus we need to use
        # the functional interface
        target = tv_transf_F.resize(target, self.tg_size, interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)
        
        if self.transform_method == "erosion":
            img = erosion(img, target)
        if self.transform_method == "skeleton":
            img = skeletization(img, target)
        elif self.transform_method == "contour_ratio":
            img = contour_ratio(img, target, self.variation)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)
        target = target[0]

        return img, target

class ValidTransforms:

    def __init__(self, tg_size):
        
        self.tg_size = tg_size

        to_dtype = tv_transf.ToDtype(
            {
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64
            },
            scale=True   # Mask is not scaled
        )

        unwrap = tv_transf.ToPureTensor()

        self.transform = tv_transf.Compose((
            to_dtype,
            unwrap
        ))

    def __call__(self, img, target):
        
        img = tv_transf_F.resize(img, self.tg_size)
        # NEAREST_EXACT has a 0.01 better Dice score than NEAREST. The
        # object oriented version of resize uses NEAREST, thus we need to use
        # the functional interface
        target = tv_transf_F.resize(target, self.tg_size, interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)
        target = target[0]

        return img, target

class TestTransforms:

    def __init__(self, tg_size):

        self.tg_size = tg_size

        to_dtype = tv_transf.ToDtype(
            torch.float32,
            scale=True
        )

        unwrap = tv_transf.ToPureTensor()

        self.transform = tv_transf.Compose((
            to_dtype,
            unwrap
        ))

    def __call__(self, img):

        img = tv_tensors.Image(img)
        img = tv_transf_F.resize(img, self.tg_size)
        img = self.transform(img)

        return img

def get_train_val_datasets(csv_path_train, csv_path_val, tg_size=(512, 512), label_values=(0, 255), channels='all', transform = None,  variation = None):

    train_dataset = TrainDataset(
        csv_path=csv_path_train, label_values=label_values, channels=channels
        )
    val_dataset = TrainDataset(
        csv_path=csv_path_val, label_values=label_values, channels=channels)

    train_dataset.transforms = TrainTransforms(tg_size, transform, variation)
    val_dataset.transforms = ValidTransforms(tg_size)

    return train_dataset, val_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), label_values=(0, 255), channels='all', num_workers=0, transform = None, variation = None):
    train_dataset, val_dataset = get_train_val_datasets(
        csv_path_train, csv_path_val, tg_size=tg_size, label_values=label_values, channels=channels, transform = transform, variation = variation
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_dataset(data_path, csv_path='test.csv', tg_size=(512, 512), channels='all'):
    # csv_path will only not be test.csv when we want to build training set predictions
    path_test_csv = osp.join(data_path, csv_path)
    test_dataset = TestDataset(csv_path=path_test_csv, tg_size=tg_size, channels=channels)

    return test_dataset
