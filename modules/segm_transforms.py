import cv2
import random
import tensorflow as tf
import numpy as np
import collections

cv2.setNumThreads(0)


def img_size(image: np.ndarray):
    """
    Return images width and height.
    :param image: nd.array with image
    :return: width, height
    """
    return (image.shape[1], image.shape[0])


def img_crop(img, box):
    img_new = img[box[1]:box[3], box[0]:box[2]]
    return img_new


def img_saturate(img):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


class MaskToTensor:
    def __init__(self, add_background=False):
        self.add_background = add_background

    def __call__(self, mask):
        mask[mask > 0] = 1
        if self.add_background:
            background_mask = np.ones_like(mask) - mask
            mask = np.stack([mask, background_mask], axis=2)
        mask = tf.convert_to_tensor(mask)
        mask = tf.dtypes.cast(mask, 'float32')
        if len(mask.shape) < 3:
            mask = tf.expand_dims(mask, 2)
        return mask


class UseWithProb:
    """Apply a given transform with probability or return input unchanged."""
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask=None):
        if self.prob > 0 and random.uniform(0, 1) < self.prob:
            image, mask = self.transform(image, mask)
        return image, mask


class OutputTransform:
    def __init__(self, segm_thresh=0.5):
        self.segm_thresh = segm_thresh

    def __call__(self, mask):
        mask = mask > self.segm_thresh
        return mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        if mask is None:
            for trns in self.transforms:
                image = trns(image)
            return image
        else:
            for trns in self.transforms:
                image, mask = trns(image, mask)
            return image, mask


class Scale(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img, mask=None):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask


class RandomCrop(object):
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, img, mask=None):
        w, h = img_size(img)
        tw, th = int(self.scale*w), int(self.scale*h)

        # Top-left corner
        x1 = random.randint(0, tw)
        y1 = random.randint(0, th)

        # Bottom-right corner
        x2 = random.randint(w-tw, w)
        y2 = random.randint(h-th, h)
        img = img_crop(img, (x1, y1, x2, y2))
        if mask is not None:
            mask = img_crop(mask, (x1, y1, x2, y2))
        return img, mask


class SquareCrop(object):
    def __call__(self, img, mask=None):
        w, h = img_size(img)
        if w > h:
            shift = int((w-h)/2)
            box = (shift, 0, shift+h, h)
        else:
            shift = int((h-w)/2)
            box = (0, shift, w, shift+w)

        img = img_crop(img, box)
        if mask is not None:
            mask = img_crop(mask, box)
        return img, mask


def generate_new_crop(x, y, w, h, image_height, image_width,
                      width_limit=250, height_limit=125):

    start_horizontal = max(0, x - width_limit)
    new_x = random.randint(start_horizontal, x)
    start_vertical = max(0, y - height_limit)
    new_y = random.randint(start_vertical, y)
    finish_horizontal = min(image_width, x + w + width_limit)
    new_w_x = random.randint(x + w, finish_horizontal)
    finish_vertical = min(image_height, y + h + height_limit)
    new_h_y = random.randint(y + h, finish_vertical)
    if new_h_y - new_y > new_w_x - new_x\
            and new_x + new_h_y - new_y < image_width:
        new_w_x = new_x + new_h_y - new_y
    return new_x, new_y, new_w_x, new_h_y


class RandomMaskCrop(object):
    def __init__(self, width_limit=250, height_limit=125):
        self.width_limit = width_limit
        self.height_limit = height_limit

    def __call__(self, img, mask):
        height, width, channels = img.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            random_contour_id = random.randint(0, len(contours)-1)
            x, y, w, h = cv2.boundingRect(contours[random_contour_id])
            new_x, new_y, new_w_x, new_h_y = generate_new_crop(
                x, y, w, h, height, width, self.width_limit, self.height_limit)
            img = img[new_y:new_h_y, new_x:new_w_x, :]
            mask = mask[new_y:new_h_y, new_x:new_w_x]

        return img, mask


def central_crop(img, mask, part=0.1):
    h, w, c = img.shape
    img = img[int(part * h):h - int(part * h),
              int(part * w):w - int(part * w), :]
    mask = mask[int(part * h):h - int(part * h),
                int(part * w):w - int(part * w)]
    return img, mask


class RandomRotation(object):
    def __init__(self, ang_range=15, crop_part=0.1, probability=0.1):
        self.ang_range = ang_range
        self.crop_part = crop_part

    def __call__(self, img, mask):
        ang_rot = random.uniform(-self.ang_range, self.ang_range)
        rows, cols, ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)
        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        mask = cv2.warpAffine(mask, Rot_M, (cols, rows))
        img, mask = central_crop(img, mask, self.crop_part)
        return img, mask


class Flip(object):
    def __init__(self, flip_code):
        self.flip_code = flip_code

    def __call__(self, imgs, trgs_mask):
        flip_imgs = cv2.flip(imgs, self.flip_code)
        trgs_mask = cv2.flip(trgs_mask, self.flip_code)
        return flip_imgs, trgs_mask


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class ToTensorColor(object):
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        tensor = tf.convert_to_tensor(img)
        tensor = tf.dtypes.cast(tensor, 'float32')
        return tf.divide(tensor, 255.0)


class AugmentImage(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, img, mask=None):
        random_gamma = random.uniform(self.gamma_low, self.gamma_high)
        random_brightness = random.uniform(
            self.brightness_low, self.brightness_high)
        random_colors = np.array(
            [random.uniform(self.color_low, self.color_high)
             for _ in range(3)]) * random_brightness

        img = img.astype(np.float)
        # randomly shift gamma
        img = img ** random_gamma
        # randomly shift brightness and color
        for i in range(3):
            img[:, :, i] = img[:, :, i] * random_colors[i]
        # saturate
        img = img_saturate(img)
        return img, mask


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    """
    def __init__(self, max_ksize=5, sigma_x=35):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, img, mask=None):
        kernal_size = (2 * random.randint(0, self.max_ksize) + 1,
                       2 * random.randint(0, self.max_ksize) + 1)
        img = cv2.GaussianBlur(img, kernal_size, self.sigma_x)
        return img, mask


class BasicNoise:
    """Apply Gauss or speckle noise to an image.

    Args:
        sigma_sq (float): Sigma squared to generate a noise matrix
        speckle (bool): False - Gauss noise, True - speckle
    """
    def __init__(self, sigma_sq, speckle=False):
        self.sigma_sq = sigma_sq
        self.speckle = speckle

    def __call__(self, img, mask=None):
        if self.sigma_sq > 0.0:
            w, h, c = img.shape
            sigma_to_use = random.uniform(0, self.sigma_sq)
            gauss = np.random.normal(0, sigma_to_use, (w, h, c))
            img = img.astype(np.int32)
            if self.speckle:
                img = img * gauss
            else:
                img = img + gauss
            img = img_saturate(img)
        return img, mask


class ComposeSegDet(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, trgs_mask):

        if trgs_mask is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, trgs_mask = t(img, trgs_mask)
            return img, trgs_mask


def train_transforms(dataset='coco', scale_size=(512, 512), sigma_g=25,
                     ang_range=15, width_limit=250, height_limit=150,
                     augment_params=[0.8, 1.2, 0.8, 1.2, 0.8, 1.2],
                     crop_scale=0.2, add_background=False):
    transforms_dict = dict()
    if dataset != 'cityscapes':
        transforms_dict['transform'] = ComposeSegDet([
            UseWithProb(RandomRotation(ang_range), 0.5),
            RandomCrop(crop_scale),
            SquareCrop(),
            Scale(scale_size),
            UseWithProb(HorizontalFlip(), 0.5),
            UseWithProb(AugmentImage(augment_params), 0.5),
            UseWithProb(RandomGaussianBlur(), 0.2),
            UseWithProb(BasicNoise(sigma_g), 0.3)
        ])
    else:
        transforms_dict['transform'] = ComposeSegDet([
            UseWithProb(RandomRotation(ang_range), 0.5),
            RandomMaskCrop(width_limit, height_limit),
            SquareCrop(),
            Scale(scale_size),
            UseWithProb(HorizontalFlip(), 0.5),
            UseWithProb(AugmentImage(augment_params), 0.5),
            UseWithProb(RandomGaussianBlur(), 0.2),
            UseWithProb(BasicNoise(sigma_g), 0.3)
        ])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = MaskToTensor(
        add_background=add_background)
    return transforms_dict


def test_transforms(dataset='coco', scale_size=(512, 512),
                    add_background=False):
    transforms_dict = dict()
    if dataset != 'cityscapes':
        transforms_dict['transform'] = ComposeSegDet([
            SquareCrop(),
            Scale(scale_size)
        ])
    else:
        transforms_dict['transform'] = ComposeSegDet([
            RandomMaskCrop(0, 0),
            SquareCrop(),
            Scale(scale_size)
        ])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = MaskToTensor(
        add_background=add_background)
    return transforms_dict


def convert_transforms(scale_size=(512, 512)):
    transforms_dict = dict()
    transforms_dict['transform'] = ComposeSegDet([
        SquareCrop(),
        Scale(scale_size)
    ])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = None
    return transforms_dict
