import cv2
import random
import torch
import os.path as osp
from glob import glob


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

# used in project
def unpaired_random_crop(img_hrs, img_lr_ns, img_lr_cs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_hrs (list[ndarray] | ndarray | list[Tensor] | Tensor): clean HR images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lr_ns (list[ndarray] | ndarray): noisy LR images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lr_cs (list[ndarray] | ndarray): bicubic downsampled HR images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): HR patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_hrs, list):
        img_hrs = [img_hrs]
    if not isinstance(img_lr_ns, list):
        img_lr_ns = [img_lr_ns]
    if not isinstance(img_lr_cs, list):
        img_lr_cs = [img_lr_cs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_hrs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lr, w_lr = img_lr_ns[0].size()[-2:]
        h_hr, w_hr = img_hrs[0].size()[-2:]
    else:
        h_lr, w_lr = img_lr_ns[0].shape[0:2]
        h_hr, w_hr = img_hrs[0].shape[0:2]
    
    lq_patch_size = gt_patch_size // scale

    # if h_hr != h_lr * scale or w_hr != w_lr * scale:
    #     raise ValueError(f'Scale mismatches. HR ({h_hr}, {w_hr}) is not {scale}x multiplication of LR ({h_lr}, {w_lr}).')
    if h_lr < lq_patch_size or w_lr < lq_patch_size:
        raise ValueError(f'LR ({h_lr}, {w_lr}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for noisy lr patch
    top = random.randint(0, h_lr - lq_patch_size)
    left = random.randint(0, w_lr - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lr_ns = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lr_ns]
    else:
        img_lr_ns = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lr_ns]

    # randomly choose top and left coordinates for clean lr patch
    top = random.randint(0, h_hr // scale - lq_patch_size)
    left = random.randint(0, w_hr // scale - lq_patch_size)

    # crop lr patch
    if input_type == 'Tensor':
        img_lr_cs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lr_cs]
    else:
        img_lr_cs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lr_cs]

    # randomly choose top and left coordinates for hr patch
    # top_gt = random.randint(0, h_hr - lq_patch_size * scale)
    # left_gt = random.randint(0, w_hr - lq_patch_size * scale)
    
    # choose top and left according to clean lr patch
    top_gt, left_gt = int(top * scale), int(left * scale)

    # crop hr patch
    if input_type == 'Tensor':
        img_hrs = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_hrs]
    else:
        img_hrs = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_hrs]

    if len(img_hrs) == 1:
        img_hrs = img_hrs[0]
    if len(img_lr_ns) == 1:
        img_lr_ns = img_lr_ns[0]
    if len(img_lr_cs) == 1:
        img_lr_cs = img_lr_cs[0]

    return img_hrs, img_lr_ns, img_lr_cs

# used in project
def paired_random_crop(img_hrs, img_lr_ns, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_hrs (list[ndarray] | ndarray | list[Tensor] | Tensor): clean HR images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lr_ns (list[ndarray] | ndarray): noisy LR images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): HR patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_hrs, list):
        img_hrs = [img_hrs]
    if not isinstance(img_lr_ns, list):
        img_lr_ns = [img_lr_ns]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_hrs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lr, w_lr = img_lr_ns[0].size()[-2:]
        h_hr, w_hr = img_hrs[0].size()[-2:]
    else:
        h_lr, w_lr = img_lr_ns[0].shape[0:2]
        h_hr, w_hr = img_hrs[0].shape[0:2]
    
    lq_patch_size = gt_patch_size // scale

    if h_hr != h_lr * scale or w_hr != w_lr * scale:
        raise ValueError(f'Scale mismatches. HR ({h_hr}, {w_hr}) is not {scale}x multiplication of LR ({h_lr}, {w_lr}).')
    if h_lr < lq_patch_size or w_lr < lq_patch_size:
        raise ValueError(f'LR ({h_lr}, {w_lr}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for noisy lr patch
    top = random.randint(0, h_lr - lq_patch_size)
    left = random.randint(0, w_lr - lq_patch_size)
    
    # 2022-02-10 18:46:25 modified, comply with aligned which has black boarder
    #       carefully do not crop in the boarder part
    # boarder_max = 4 + scale
    # top = random.randint(boarder_max, h_lr - lq_patch_size - boarder_max)
    # left = random.randint(boarder_max, w_lr - lq_patch_size - boarder_max)


    # crop lq patch
    if input_type == 'Tensor':
        img_lr_ns = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lr_ns]
    else:
        img_lr_ns = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lr_ns]
    
    # choose top and left according to lq patch
    top_gt, left_gt = int(top * scale), int(left * scale)

    # crop hr patch
    if input_type == 'Tensor':
        img_hrs = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_hrs]
    else:
        img_hrs = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_hrs]

    if len(img_hrs) == 1:
        img_hrs = img_hrs[0]
    if len(img_lr_ns) == 1:
        img_lr_ns = img_lr_ns[0]

    return img_hrs, img_lr_ns


# used in project
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
