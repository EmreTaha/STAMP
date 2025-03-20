import torch
from torchvision import transforms
import monai.transforms
from monai.config import KeysCollection
from typing import Union
import numpy as np

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def bscan_sup_transformsv4(NORM,rotation=10,antialias=True):
    """
        Applies a series of transformations to an image for training purposes.
        Args:
            NORM (tuple): A tuple containing the mean and standard deviation for normalization.
            rotation (int, optional): The degree of rotation for the RandomRotation transformation. Default is 10.
            antialias (bool, optional): Whether to apply antialiasing. Default is True. Currently it has no effect
        Returns:
            torchvision.transforms.Compose: A composed transform that applies random affine transformations, 
            random rotations, horizontal flips, converts image dtype to float32, and normalizes the image.
    """

    train_transform = transforms.Compose([transforms.RandomAffine(0,(0.05,0.05),fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ConvertImageDtype(torch.float32),
                                      transforms.Normalize(*NORM)])
    return train_transform

# SimCLR Transformations
def simclr_transforms(image_size: Union[int, tuple], jitter: tuple = (0.4, 0.4, 0.2, 0.1),
                      p_blur: float = 1.0, p_solarize: float = 0.0,
                      normalize: list = [[0.485],[0.229]], translation=True, scale=(0.4, 0.8), gray_scale=True,
                      sol_threshold=0.42, p_jitter=0.8, vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    """
    Returns a composition of transformations for SimCLR training.

    Args:
        image_size (Union[int, tuple]): The size of the output image. If int, the output image will be square with sides of length `image_size`. If tuple, the output image will have dimensions `image_size[0]` x `image_size[1]`.
        jitter (tuple, optional): Tuple of four floats representing the range of random color jitter. Defaults to (0.4, 0.4, 0.2, 0.1).
        p_blur (float, optional): Probability of applying Gaussian blur. Defaults to 1.0.
        p_solarize (float, optional): Probability of applying random solarization. Defaults to 0.0.
        normalize (list, optional): List of two lists representing the mean and standard deviation for image normalization. Defaults to [[0.485],[0.229]].
        translation (bool, optional): Whether to apply small random translation. Defaults to True.
        scale (tuple, optional): Tuple of two floats representing the range of random scale for random resized crop. Defaults to (0.4, 0.8).
        gray_scale (bool, optional): Whether the input image is in gray scalr or not. Defaults to True.
        sol_threshold (float, optional): Threshold for random solarization. Defaults to 0.42.
        p_jitter (float, optional): Probability of applying color jitter. Defaults to 0.8.
        vertical_flip (bool, optional): Whether to apply random vertical flip. Defaults to False.
        p_horizontal_flip (float, optional): Probability of applying random horizontal flip. Defaults to 0.5.
        rotate (bool, optional): Whether to apply random rotation. Defaults to False.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    trans_list = []
    image_size = pair(image_size)

    if translation:
        trans_list.append(transforms.RandomAffine(0,(0.05,0.05), fill=0, interpolation=transforms.InterpolationMode.BILINEAR))
    if rotate:
        trans_list.append(transforms.RandomRotation(rotate, interpolation=transforms.InterpolationMode.BILINEAR))

    trans_list += [transforms.RandomResizedCrop(image_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), #TODO antialias is introduced on 0.15 not stable yet
                  transforms.RandomHorizontalFlip(p=p_horizontal_flip),
                  transforms.ConvertImageDtype(torch.float32)]


    trans_list.append(transforms.RandomApply([transforms.ColorJitter(*jitter)], p=p_jitter))

    if vertical_flip:
        trans_list.append(transforms.RandomVerticalFlip(p=0.5))
    #If image is not grayscale add RandomGrayscale
    if not gray_scale:
        trans_list.append(transforms.RandomGrayscale(p=0.2))
    # Turn off blur for small images
    if image_size[0]<=32:
        p_blur = 0.0
    # Add Gaussian blur
    if p_blur==1.0:
        trans_list.append(transforms.GaussianBlur(image_size[0]//20*2+1))
    elif p_blur>0.0:
        trans_list.append(transforms.RandomApply([transforms.GaussianBlur(image_size[0]//20*2+1)], p=p_blur))
    # Add RandomSolarize
    if p_solarize>0.0:
        trans_list.append(transforms.RandomSolarize(sol_threshold, p=p_solarize))

    if normalize:
        trans_list.extend([transforms.Normalize(*normalize)])
    
    return transforms.Compose(trans_list)

# A wrapper that performs returns two augmented images
class TwoTransform(object):
    """Applies data augmentation two times."""

    def __init__(self, base_transform, sec_transform = None, temporal=False):
        self.base_transform = base_transform
        self.sec_transform = base_transform if sec_transform is None else sec_transform
        self.temporal = temporal

    def __call__(self, x):
        x1 = self.base_transform(x)
        if self.temporal:
            return x1
        x2 = self.sec_transform(x)
        return x1,x2

class VICReg_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=[[0.485],[0.229]], gray_scale=True, temporal=False, translation=True, scale=(0.4, 0.8)):
        trans1 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale)
        
        trans2 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale)
        
        super().__init__(trans1, trans2, temporal=temporal)

def Barlow_augmentaions(image_size, normalize=[[0.485],[0.229]], gray_scale=True, translation=False, p_blur=0.8, scale=(0.08, 1.0), 
                        temporal=False, sol_threshold=0.42, jitter: tuple = (0.4, 0.4, 0.2, 0.1), p_solarize=0.2, p_jitter=0.8, 
                        vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    if temporal:        
        trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        return trans1
    
    trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = 0.0,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    trans2 = simclr_transforms(image_size,
                                   p_blur = max(0.0, 1.0-p_blur),
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter=p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    return [trans1, trans2]

def Monai3DTransformsv3(transforms, image_keys=["image"],locs=(32,96), octstep=2, spatial_size=(224,224), interp_mode="area"):
    '''
    This function creates a series of transformations for 3D OCT data.
    
    args:
        transforms (list): A list of torchviion transformations to be applied to the input image.
        image_keys (list): A list of keys corresponding to the input image.
        locs (tuple): A tuple containing the start and end locations for the B-scans.
        octstep (int): The step size between B-scans.
        spatial_size (tuple): The size of the output image.
        interp_mode (str): The interpolation mode to be used for resizing.
    '''
    # Now first resizes, then takes slices
    # Also now default interp_mode is area
    transf = monai.transforms.Compose([monai.transforms.Lambdad(keys =image_keys, func = lambda x: torch.from_numpy(x).unsqueeze(0)),  #unsqueeze is necessary for torch vision, we have 1 channel!
                     monai.transforms.Resized(keys = image_keys, mode = interp_mode,spatial_size=(spatial_size[0],spatial_size[1],-1),anti_aliasing=True),# 1 channel!, permute is needed because torchvision depth is first
                     SliceWithFovead(keys = image_keys+['fovea'], allow_missing_keys=False, locs=locs, octstep=octstep),
                     monai.transforms.Lambdad(keys = image_keys, func = lambda x: x.permute(3,0,1,2)), # Because torch vision follows ...,C,H,W
                     monai.transforms.RandLambdad(keys = image_keys, func = lambda x: transforms(x),prob=1.0),
                     monai.transforms.Lambdad(keys = image_keys, func = lambda x: x.permute(1,0,2,3)), # Revert back to ...,C,D,H,W
                     monai.transforms.ToTensord(keys = image_keys + ["label"], track_meta=False)])
    return transf

class SliceWithFovead(monai.transforms.MapTransform):
    '''
    This class slices the input image with respect to the fovea position.
    
    args:
        keys (list): A list of keys corresponding to the input image.
        allow_missing_keys (bool): Whether missing keys are allowed.
        locs (tuple): A tuple containing the start and end locations for the B-scans.
        octstep (int): The step size between B-scans.
    '''
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool, locs: tuple = (32,96), octstep: int = 2) -> None:
        super().__init__(keys, allow_missing_keys)
        self.locs = locs
        self.octstep = octstep
    def __call__(self, data):
        fov = data['fovea_pos']
        locs_diff = (self.locs[1] - self.locs[0])//2
        if not torch.is_tensor(data['image']):
            temp = torch.from_numpy(data['image'])
        else:
            temp = data['image']
        data['image'] = temp[...,fov-locs_diff:fov+locs_diff:self.octstep]
        return data

class shiftBscansv2(monai.transforms.MapTransform, monai.transforms.Randomizable):
    '''
    This class shifts the B-scans by a random amount.

    args:
        keys (list): A list of keys corresponding to the input image.
        allow_missing_keys (bool): Whether missing keys are allowed.
        shift_amount (int): The max amount by which to shift the B-scans.
        size (int): The size of the output image.
        prob (float): The probability of applying the transformation.
    '''
    # Creates a random number, and shifts bscans by this number
    # Expects D,C,H,W fuck torchvision
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool, shift_amount: int, size=32, prob: float = 1.0) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob = np.clip(prob, 0.0, 1.0)
        self.shift_amount = shift_amount
        self.size = size
    def __call__(self, data):
        if self.R.random() < self.prob:
            for key in self.keys: #self.key_iterator(data):
                item = data[key]
                rand_num = self.R.randint(-self.shift_amount,self.shift_amount) # Different shift amount for each key!
                data[key] = item[:,rand_num+self.shift_amount:self.size+rand_num+self.shift_amount,...]
        else: # Below threshold, then dont shift, take a fovea centered, input is already sliced with shift margin around fovea
            for key in self.keys: #self.key_iterator(data):
                item = data[key]
                rand_num = 0
                data[key] = item[:,rand_num+self.shift_amount:self.size+rand_num+self.shift_amount,...]
        return data

def Monai3DTwoInpContrastTransformsv3(transforms, shift_amount=5, locs=(32,96), octstep=2, shift_prob=1.0, spatial_size=(224,224), interp_mode="area"):
    '''
    This function creates a series of transformations for 3D OCT input.
    args:
        transforms (list): A list of torchviion transformations to be applied to the input image.
        shift_amount (int): The max amount by which to shift the B-scans.
        locs (tuple): A tuple containing the start and end locations for the B-scans.
        octstep (int): The step size between B-scans.
        shift_prob (float): The probability of applying the bscan shift transformation.
        spatial_size (tuple): The size of the output image.
        interp_mode (str): The interpolation mode to be used for resizing
    '''
    # Now first resizes, then takes slices
    # Also now standart interpolation is area
    transf = monai.transforms.Compose([monai.transforms.Lambdad(keys = ["image"], func = lambda x: torch.from_numpy(x).unsqueeze(0)),  #unsqueeze is necessary for torch vision, we have 1 channel!
                     monai.transforms.Resized(keys = ["image"], mode = interp_mode, spatial_size=(spatial_size[0],spatial_size[1],-1),anti_aliasing=True),
                     SliceWithFovead(keys = ["image", 'fovea_pos'], allow_missing_keys=False, locs=(locs[0]-octstep*shift_amount,locs[1]+octstep*shift_amount), octstep=octstep),
                     monai.transforms.Lambdad(keys = ["image"], func = lambda x: x.permute(0,3,1,2)),
                     shiftBscansv2(keys = ["image"], shift_amount=shift_amount, size=(locs[1]-locs[0])//octstep, allow_missing_keys=False, prob=shift_prob), # has input C,D,H,W
                     monai.transforms.Lambdad(keys = ["image"], func = lambda x: x.permute(1,0,2,3)), # Because torch vision follows ...,C,H,W 
                     monai.transforms.RandLambdad(keys = ["image"], func = lambda x: transforms(x),prob=1.0),
                     monai.transforms.Lambdad(keys = ["image"], func = lambda x: x.permute(1,0,2,3)), # Revert back to ...,C,D,H,W
                     monai.transforms.ToTensord(keys = ["image","label"], track_meta=False)])
                     
    return transf