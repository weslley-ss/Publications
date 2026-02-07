import copy

import numpy as np
from skimage.morphology import skeletonize
import torch
from torch import nn

class EffectiveReceptiveField:
    """Calculate the effective receptive field of a neural network. 

    Luo, W., Li, Y., Urtasun, R. and Zemel, R., 2016. Understanding the effective receptive 
    field in deep convolutional neural networks. Advances in neural information processing 
    systems, 29.

    https://arxiv.org/abs/1701.04128
    """

    def __init__(
            self, 
            model, 
            ds, 
            use_skeleton: bool = False, 
            device: str = "cuda"
            ):
        """
        Parameters
        ----------
        model 
            The model to analyze
        ds
            The dataset
        use_skeleton
            If True, only calculates the receptive field of pixels in the skeleton of the vessels.
        device
            The device to use
        """

        model.to(device)
        model.eval()

        labels = []
        for _, label in ds:
            if use_skeleton:
                # Store vessel skeletons instead of labels
                label = torch.from_numpy(skeletonize(label.numpy()==1))
            labels.append(label)

        self.model = model
        self.ds = ds
        self.labels = labels
        self.device = device
        
    def receptive_field_from_pixel(
            self, 
            idx: int, 
            pixel: tuple | None = None, 
            scores_norm: str | None = "sigmoid"
            ) -> np.ndarray:
        """Calculate the receptive field of a pixel in a given image.
        
        Parameters
        ----------
        idx
            The image index in the dataset
        pixel
            The output pixel to calculate the receptive field. If None, uses the pixel at the center.
        scores_norm
            Which function to apply to the output of the model. Options are "softmax" and "sigmoid".
            If None, uses the raw scores.

        Returns
        -------
        rf
            The receptive field
        """

        model = self.model
        img, _ = self.ds[idx]
        img = img.to(self.device)
        # Set requires_grad to True to calculate gradients with respect to the input
        img.requires_grad = True

        output = model(img.unsqueeze(0))[0]
        if scores_norm=="softmax":
            output = output.softmax(0)
        elif scores_norm=="sigmoid":
            output = output.sigmoid()
        # Output of the vessel class
        output = output[-1]

        if pixel is None:
            # Get pixel at the middle of the activation map
            size = output.shape[-2:]
            pixel = (size[0]//2, size[1]//2)
        pix_val = output[pixel[0], pixel[1]]
        # Calculate gradients
        pix_val.backward()

        # Gradient with respect to the input, averaged over the image channels
        rf = img.grad.abs().mean(0)

        #rf = rf/rf.max()
        rf = rf.to("cpu").numpy()

        return rf
    
    def receptive_field_from_vessels(
            self, 
            n_samples: int = 10, 
            scores_norm: str | None = "sigmoid",
            seed: int = 0
            ) -> dict:
        """Calculate the receptive field of `n_sample` vessel pixels randomly drawn from the dataset

        Parameters
        ----------
        n_samples
            Number of vessel pixels to sample. n_sample receptive fields will be calculated
        scores_norm
            Which function to apply to the output of the model. Options are "softmax" and "sigmoid".
            If None, uses the raw scores.
        seed
            Seed for the random number generator

        Returns
        -------
        rf_data
            A dictionary with the following structure:
            {
            "img_idx":<index of the image>,
            "pixel":<pixel coordinates>,
            "rf"<receptive field of the pixel>:
            }
        """

        
        torch.manual_seed(seed)

        rf_data = {}
        for rep in range(n_samples):
            # Randomly draw an image index
            img_idx = torch.randint(0, len(self.ds), (1,)).item()
            label = self.labels[img_idx]

            vessel_indices = torch.nonzero(label==1)
            n_vessels = vessel_indices.size(0)
            # Randomly draw a pixel index
            pix_idx = torch.randint(0, n_vessels, (1,)).item()
            pixel = vessel_indices[pix_idx].tolist()

            rf = self.receptive_field_from_pixel(img_idx, pixel=pixel, scores_norm=scores_norm)
        
            rf_data[rep] = {
                "img_idx": img_idx,
                "pixel": pixel,
                "rf": rf,
            }

        return rf_data

class TheoreticalReceptiveField:
    """Calculate the theoretical receptive field of a pixel in the activation map of a neural network layer. 
    """

    conv_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    conv_transp_layers = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    norm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    pool_layers = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
    linear_layers = (nn.Linear,)
    ignored_layers = (nn.Sequential, nn.ModuleDict, nn.ModuleList, nn.Identity)

    def __init__(
            self, 
            model, 
            device: str = "cuda",
            prepare_model: bool = True
            ):

        self.device = device
        model = copy.deepcopy(model)
        model.to(self.device)
        model.eval()
        self.model = model
        if prepare_model:
            self.prepare_model(model)

    def prepare_model(
            self, 
            model
            ):
        """This method changes some of the layers of the model to avoid trivial receptive fields.
        """

        # Dictionary with the relations between the layers of the model
        module_relations = {".self": (None, model)}
        for parent_name, parent_module in model.named_modules():
            for name, module in parent_module.named_children():
                module_relations[f"{parent_name}.{name}"] = (parent_module, module)

        with torch.no_grad():
            for relation, (parent_module, module) in module_relations.items():
                _, name = relation.rsplit(".")
                if isinstance(module, self.conv_layers):
                    # Set filters to 1/num_vals_filter. Assumes filters have the same
                    # size in all dimensions
                    n = module.weight[0].numel()
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.
                #elif isinstance(module, nn.ReLU):
                    #module.inplace = False
                #    pass
                elif isinstance(module, self.conv_transp_layers):
                    ks = module.kernel_size[0]
                    stride = module.stride[0]
                    dim = len(module.kernel_size)
                    # Effective number of input features is ks//stride for each dimension
                    n = (ks//stride)**dim
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.
                elif isinstance(module, self.norm_layers):
                    # Disable batchnorm
                    module.training = False
                    module.weight[:] = 1.
                    module.bias[:] = 0.
                    module.running_mean[:] = 0.
                    module.running_var[:] = 1.
                elif isinstance(module, self.pool_layers):
                    ks = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    # Change maxpool to avgpool
                    setattr(parent_module, name, nn.AvgPool2d(ks, stride, padding))
                elif isinstance(module, self.linear_layers):
                    # Number of input features
                    n = module.weight.shape[1]
                    module.weight[:] = 1./n
                elif len(list(module.parameters(recurse=False)))>0:
                    print(f"Warning, module {relation} was not recognized.")

    def receptive_field(
            self, 
            num_channels: int = 1, 
            img_size: tuple = (512, 512), 
            pixel: tuple | None = None
            ) -> np.ndarray:
        """Calculate the receptive field of an output pixel.
        
        Parameters
        ----------
        num_channels
            The number of channels of the input
        img_size
            Image size to use as input
        pixel
            Which pixel to use. If not provided, uses the pixel at the center.

        Returns
        -------
        rf
            An image containing the receptive field.
        """

        model = self.model

        x = torch.ones(
            1, num_channels, img_size[0], img_size[1], requires_grad=True, device=self.device
        )
        #x = torch.rand(
        #    1, num_channels, img_size[0], img_size[1], requires_grad=True, device=self.device
        #)
        output = model(x)

        if pixel is None:
            # Get pixel at the middle of the activation map
            size = output.shape[-2:]
            pixel = (size[0]//2, size[1]//2)
        pix_val = output[0, 0, pixel[0], pixel[1]]
        # Calculate gradients
        pix_val.backward()

        # Gradient with respect to the input
        rf = x.grad[0, 0]

        rf = rf/rf.max()
        rf = rf.to("cpu").numpy()

        return rf
    
def calculate_rfs(
        model, 
        ds, 
        n_samples: int = 10, 
        factor: float = 0.0455, 
        use_skeleton: bool = False, 
        return_rfs: bool = False,
        seed: int = 0, 
        device: str = "cuda"
        ):
    """Calculate the theoretical and effective receptive fields of a neural network."

    Parameters
    ----------
    model
        The model to analyze
    ds
        The dataset
    n_samples
        Number of vessel pixels to sample. n_sample receptive fields will be calculated
    factor
        Factor to set the threshold to calculate the effective receptive field. All pixels with a value
        greater than factor*rf[pixel] are considered to be part of the receptive field.
    use_skeleton    
        If True, only calculates the receptive field of pixels in the skeleton of the vessels.
    return_rfs
        If True, also returns the receptive fields. If False, only returns the sizes.
    seed
        Seed for the random number generator
    device
        The device to use

    Returns
    -------
    output
        A dictionary with the following structure:
        {
        "trf_size":<size of the theoretical receptive field>,
        "rf_data":<effective receptive field data>,
        "trf":<theoretical receptive field> (if return_rfs is True)
        }
    """
    
    trc_class = TheoreticalReceptiveField(model, device=device, )
    img, _ = ds[0]
    num_channels, *img_size = img.shape
    trf = trc_class.receptive_field(num_channels, img_size)
    bbox = get_bbox(trf, threshold=0)
    
    # Important! Here the size of the theoretical receptive field is calculated as the size of 
    # the bounding box. We assume that the bounding box is a square.
    trf_size = bbox[2] - bbox[0] + 1

    erf_class = EffectiveReceptiveField(model, ds, use_skeleton=use_skeleton, device=device)
    rf_data = erf_class.receptive_field_from_vessels(n_samples=n_samples, seed=seed)
    erf_sizes = get_erf_sizes(rf_data, factor)

    for k in rf_data:
        rf_data[k]['size'] = erf_sizes[k]
        if not return_rfs:
            # Remove the receptive field
            del rf_data[k]['rf']

    output = {
        "trf_size": trf_size,
        "rf_data": rf_data,
    }

    if return_rfs:
        output["trf"] = trf

    return output

def calculate_rf_stats(rf_data: dict):
    """Returns the trf size and the mean and standard deviation of the erf sizes.

    Parameters
    ----------
    rf_data
        The output of the method `calculate_rfs`

    Returns
    -------
    trf_size
        The size of the theoretical receptive field
    erf_mean
        The mean of the effective receptive field sizes
    erf_std
        The standard deviation of the effective receptive field sizes
    """

    trf_size = rf_data["trf_size"]

    erf_sizes = []
    for sample in rf_data["rf_data"].values():
        erf_sizes.append(sample["size"])
    erf_sizes = np.array(erf_sizes)

    erf_mean = erf_sizes.mean()
    erf_std = erf_sizes.std()


    return trf_size, erf_mean, erf_std

def get_erf_sizes(erf_data: dict, factor: float = 0.0455):
    """Calculate the sizes of the effective receptive fields. 

    Important! The size of the effective receptive field is calculated as the square root of the area
    of the region where the receptive field is greater than factor*rf[pixel]. pixel is the pixel where
    the receptive field was calculated.

    Parameters
    ----------
    erf_data
        A dictionary with the effective receptive field data. This dictionary is returned by the
        method `receptive_field_from_vessels` of the `EffectiveReceptiveField` class
    factor
        Factor to set the threshold to calculate the effective receptive field. All pixels with a value
        greater than factor*rf[pixel] are considered to be part of the receptive field.

    Returns
    -------
    sizes
        A list with the sizes of the effective receptive fields
    """

    sizes = []
    for i, output in erf_data.items():
        rf = output['rf']
        pixel = output['pixel']
        threshold = factor*rf[pixel[0], pixel[1]]

        area = (rf>threshold).sum().item()

        sizes.append(area**0.5)

    return sizes

def get_bbox(rf, threshold=1e-8):
    """Get the bounding box of the receptive field."

    Parameters
    ----------
    rf
        The receptive field
    threshold
        Threshold to calculate the bounding box

    Returns
    -------
    bbox
        The bounding box of the receptive field in the format (r0, c0, r1, c1)
    """

    inds = np.argwhere(rf>threshold)
    r0, c0 = inds.min(axis=0)
    r1, c1 = inds.max(axis=0)
    r0, c0, r1, c1 = r0.item(), c0.item(), r1.item(), c1.item()
    bbox = (r0, c0, r1, c1)

    return bbox
