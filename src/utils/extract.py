import os
import imageio
import moviepy.editor as mpy
import numpy as np
import cv2

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import One
import mindspore.ops as ops


def make_video(array, fps, filename):
    # encode sequence of images into gif string
    if isinstance(array, Tensor):
        array = array.asnumpy()
    clip = mpy.ImageSequenceClip(list(array), fps=fps)

    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)


def generate_images(opt):
    for exp_dir in opt.experiments:
        fakes_path = os.path.join(opt.saver.eval_dir, 'random_samples.npy')
        print(fakes_path)
        os.makedirs(os.path.join(opt.saver.eval_dir, opt.save_path), exist_ok=True)
        print('Generating dir {}'.format(os.path.join(opt.saver.eval_dir, opt.save_path)))

        random_samples = Tensor(np.load(fakes_path))
        random_samples = ops.Transpose()(random_samples, (0, 2, 3, 1))[:opt.max_samples]
        random_samples = (random_samples + 1) / 2
        random_samples = random_samples[:20] * 255
        random_samples = (random_samples.asnumpy()).astype(np.uint8)
        for i, sample in enumerate(random_samples):
            imageio.imwrite(os.path.join(opt.saver.eval_dir, opt.save_path, f'fake_{i}.png'), sample)


def generate_gifs(opt):
    for exp_dir in opt.experiments:
        reals_path = os.path.join(opt.saver.eval_dir, 'real_full_scale.npy')
        fakes_path = os.path.join(opt.saver.eval_dir, 'random_samples.npy')
        os.makedirs(os.path.join(opt.saver.eval_dir, opt.save_path), exist_ok=True)
        print('Generating dir {}'.format(os.path.join(opt.saver.eval_dir, opt.save_path)))

        real_sample = np.load(reals_path)
        make_video(real_sample, 4, os.path.join(opt.saver.eval_dir, opt.save_path, 'real.gif'))

        random_samples = Tensor(np.load(fakes_path)).transpose(0, 2, 3, 4, 1)[:opt.max_samples]

        # Make grid
        grid_image = cv2.hconcat(real_sample)
        imageio.imwrite(os.path.join(opt.saver.eval_dir, opt.save_path, 'real_unfold.png'), grid_image)

        fake = (random_samples.asnumpy() * 255).astype(np.uint8)
        fake_transpose = Tensor(fake).transpose(0, 1, 4, 2, 3)[:, ::2]  # BxTxCxHxW
        fake_reshaped = np.array([fake_transpose[b][t].asnumpy() for b in range(fake_transpose.shape[0]) \
                                  for t in range(fake_transpose.shape[1])])  # (BxT)xCxHxW
        fake_reshaped = fake_reshaped[:10 * fake_transpose.shape[1]].transpose(0, 2, 3, 1)
        grid_frames = np.array([cv2.hconcat(fake_reshaped[b:b + fake_transpose.shape[1]])
                                for b in range(fake_transpose.shape[0])])
        grid_image = cv2.vconcat(grid_frames)
        imageio.imwrite(os.path.join(opt.saver.eval_dir, opt.save_path, 'fake_unfold.png'), grid_image)

        white_space = Tensor(shape=random_samples.shape, init=One(), dtype=mstype.float32)[:, :, :, :10] * 255

        random_samples = random_samples.asnumpy()
        random_samples = (random_samples * 255).astype(np.uint8)
        white_space = white_space.asnumpy()
        white_space = (white_space * 255).astype(np.uint8)

        concat_gif = []
        for i, (vid, ws) in enumerate(zip(random_samples, white_space)):
            if i < len(random_samples) - 1:
                concat_gif.append(np.concatenate((vid, ws), axis=2))
            else:
                concat_gif.append(vid)
        concat_gif = np.concatenate(concat_gif, axis=2)
        make_video(concat_gif, 4, os.path.join(opt.saver.eval_dir, opt.save_path, 'fake.gif'))