import os
import imageio
import moviepy.editor as mpy
import numpy as np
from cv2 import vconcat

import mindspore
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import One
import mindspore.ops as ops


def make_video(tensor, fps, filename):
    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

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
        print('Generating dir {}'.format(os.path.join(exp_dir, opt.save_path)))

        with open(fakes_path, 'rb') as f:
            random_samples = Tensor(np.load(f))
        random_samples = ops.Transpose()(random_samples, (0, 2, 3, 1))[:opt.max_samples]
        random_samples = (random_samples + 1) / 2
        random_samples = random_samples[:20] * 255
        random_samples = (random_samples.asnumpy()).astype(np.uint8)
        for i, sample in enumerate(random_samples):
            imageio.imwrite(os.path.join(opt.saver.eval_dir, opt.save_path, f'fake_{i}.png'), sample)


def generate_gifs(opt):
    for exp_dir in opt.experiments:
        reals_path = os.path.join(exp_dir, 'real_full_scale.pth')
        fakes_path = os.path.join(exp_dir, 'random_samples.pth')
        os.makedirs(os.path.join(exp_dir, opt.save_path), exist_ok=True)
        print('Generating dir {}'.format(os.path.join(exp_dir, opt.save_path)))

        real_sample = mindspore.load(reals_path)
        make_video(real_sample, 4, os.path.join(exp_dir, opt.save_path, 'real.gif'))

        with open(fakes_path, 'rb') as f:
            random_samples = Tensor(np.load(f))
        random_samples = ops.Transpose()(random_samples, (0, 2, 3, 4, 1))[:opt.max_samples]

        # Make grid
        real_transpose = ops.Transpose()(Tensor(real_sample), (0, 3, 1, 2))[::2]  # TxCxHxW
        grid_image = vconcat(real_transpose).transpose(1, 2, 0)
        imageio.imwrite(os.path.join(exp_dir, opt.save_path, 'real_unfold.png'), grid_image)

        fake = (random_samples.data.cpu().numpy() * 255).astype(np.uint8)
        fake_transpose = Tensor(fake).permute(0, 1, 4, 2, 3)[:, ::2]  # BxTxCxHxW
        fake_reshaped = fake_transpose.flatten(0, 1)  # (B+T)xCxHxW
        grid_image = vconcat(fake_reshaped.asnumpy()[:10 * fake_transpose.shape[1]]).transpose(1, 2, 0)
        imageio.imwrite(os.path.join(exp_dir, opt.save_path, 'fake_unfold.png'), grid_image)

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
        make_video(concat_gif, 4, os.path.join(exp_dir, opt.save_path, 'fakes.gif'))