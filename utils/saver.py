import glob
import os
import cv2
import json
import mindspore
import numpy as np


def write_video(array, filename, opt):
    _, num_frames, height, width = array.shape
    FPS = opt.fps
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            float(FPS), (width, height))
    for i in range(num_frames):
        frame = (array[:, i, :, :] + 1) * 127.5
        frame = frame.transpose(1, 2, 0)
        video.write(np.uint8(frame))
    video.release()


class DataSaver:
    def __init__(self, opt, run_id=None):
        self.opt = opt
        if not hasattr(opt, 'experiment_dir') or not os.path.exists(opt.experiment_dir):
            clip_name = '.'.join(opt.image_path.split('/')[-1].split('.')[:-1])
            self.directory = os.path.join('run', clip_name, opt.checkname)
            if run_id is None:
                self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
                run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        else:
            self.experiment_dir = opt.experiment_dir

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.eval_dir = os.path.join(self.experiment_dir, "eval")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        self.image_dir = os.path.join(self.eval_dir, "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.iteration = 0

    def save_checkpoint(self, cell, filename='checkpoint.ckpt'):
        filename = os.path.join(self.experiment_dir, filename)
        mindspore.save_checkpoint(cell, filename)

    def load_checkpoint(self, filename):
        filename = os.path.join(self.experiment_dir, filename)
        return mindspore.load_checkpoint(filename)

    def save_json(self, obj, filename):
        filename = os.path.join(self.experiment_dir, filename)
        with open(filename,'w+') as f:
            json.dump(obj, f)

    def load_json(self, filename):
        filename = os.path.join(self.experiment_dir, filename)
        with open(filename,'r+') as f:
            obj = json.load(f)
        return obj

    def save_image(self, img, filename):
        filename = os.path.join(self.image_dir, filename)
        img = img.asnumpy().squeeze().astype(np.uint8)
        if img.ndim != 3:
            return
        if cv2.imwrite(filename, img):
            print(f'Saving image Successfully at {filename}.')

    def save_video(self, array, filename):
        filename = os.path.join(self.eval_dir, filename)
        write_video(array, filename, self.opt)
        print(f'Saving video Successfully at {filename}.')