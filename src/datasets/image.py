import random
import numpy as np
import imageio
import cv2
import logging
import os
from mindspore.dataset.vision import Normalize
import mindspore.dataset as ds

from .. import utils


class SingleImageDataset:
    def __init__(self, opt, transforms=None):
        self.zero_scale_frames = None
        self.frames = None

        self.transforms = transforms

        self.image_path = opt.image_path
        if not os.path.exists(opt.image_path):
            logging.error("invalid path")
            exit(0)

        # Get original frame size and aspect-ratio
        self.image_full_scale = imageio.imread(self.image_path)[:, :, :3]   # HWC
        self.org_size = [self.image_full_scale.shape[0], self.image_full_scale.shape[1]]
        h, w = self.image_full_scale.shape[:2]
        opt.ar = h / w  # H2W

        self.opt = opt

    def __len__(self):
        return self.opt.data_rep

    def __getitem__(self, idx):
        hflip = random.random() < 0.5 if self.opt.hflip else False

        images = self.generate_image(self.opt.scale_idx)
        images = np.array(images).transpose(2, 0, 1).astype(np.float32) \
                 if images.ndim == 3 else \
                 np.array(images).transpose(0, 3, 1, 2).astype(np.float32)
        images = images / 255  # Set range [0, 1]
        images_transformed = self._get_transformed_images(images, hflip)

        # Extract o-level index
        if self.opt.scale_idx > 0:
            images_zero_scale = self.generate_image(0)
            images_zero_scale = np.array(images_zero_scale).transpose(2, 0, 1).astype(np.float32) \
                                if images_zero_scale.ndim == 3 else \
                                np.array(images_zero_scale).transpose(0, 3, 1, 2).astype(np.float32)
            images_zero_scale = images_zero_scale / 255
            images_zero_scale_transformed = self._get_transformed_images(images_zero_scale, hflip)

            return images_transformed, images_zero_scale_transformed

        return images_transformed, np.zeros_like(images_transformed)

    @staticmethod
    def _get_transformed_images(images, hflip):
        images_transformed = images

        if hflip:
            images_transformed = np.flip(images_transformed, -1)
        # Normalize
        images_transformed = Normalize([0.5], [0.5])(images_transformed)

        return images_transformed

    def generate_image(self, scale_idx):
        base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor,
                                              self.opt.stop_scale, self.opt.img_size)
        scaled_size = [int(base_size * self.opt.ar), base_size]
        self.opt.scaled_size = scaled_size
        img = cv2.resize(self.image_full_scale, tuple(scaled_size[::-1]))
        return img


if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.nfc = 64
            self.nc_im = 3
            self.ker_size = 3
            self.num_layer = 5
            self.latent_dim = 128
            self.enc_blocks = 2
            self.padd_size = 1
            self.image_path = './data/imgs/air_balloons.jpg'
            self.hflip = True
            self.img_size = 256
            self.scale_factor = 0.75
            self.stop_scale = 9
            self.scale_idx = 2
            self.batch_size = 2
            self.data_rep = 1000

    opt = Opt()
    # 实例化数据集类
    dataset_generator = SingleImageDataset(opt)
    dataset = ds.GeneratorDataset(dataset_generator, ['data1', 'data2'])
    dataset = dataset.batch(opt.batch_size)
    dataset = dataset.shuffle(4)
    dl = dataset.create_tuple_iterator()
    # 打印数据条数
    print(dataset_generator[500][0].shape, dataset_generator[500][0].shape)