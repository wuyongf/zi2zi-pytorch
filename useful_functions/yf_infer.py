import imageio

from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import os
import argparse
import torch
import random
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import time
from model.model import chk_mkdir

from ttf2img import get_ground_truth_img

writer_dict = {'蔡云汉简体行书书法字体': 0, '方正字迹-邱氏粗瘦金书简体': 1, '方正祥隶简体': 2, '柳公权柳体': 3, '汉仪瘦金书简': 4}

if __name__ == '__main__':

    # todo: for each font(label) --- 100和
    #  1. get ground truth
    #  2. get infer_images
    #  3. get infer_gif

    # global writer_dict
    writer_dict_inv = {v: k for k, v in writer_dict.items()}

    # global - for ground truth image
    src_font_name = '仓耳今楷03-W03.ttf'
    ch = '和'

    # global - for infer: start/end checkpoint
    checkpoint_start = 500
    checkpoint_end = 25500
    checkpoint_step = 500

    for label_no in range(0,5,1):
        label = label_no
        # if(label == 1): break

        # define saving address
        save_addr = './result/'
        folder_name = str(label) + '_' + writer_dict_inv[label]

        ground_truth_addr = save_addr + folder_name + '/ground_truth/'
        images_addr = save_addr + folder_name + '/infer_images/'
        gif_addr = save_addr + folder_name + '/infer_gif/'

        os.makedirs(ground_truth_addr, exist_ok=True)
        os.makedirs(images_addr, exist_ok=True)
        os.makedirs(gif_addr, exist_ok=True)

        # # save ground truth
        # src_font_addr = '../fonts/source/' + src_font_name
        # dst_font_addr = '../fonts/target/' + writer_dict_inv[label] + '.ttf'
        # get_ground_truth_img(0,src_font_addr,dst_font_addr,label,ch,ground_truth_addr)
        #
        # # save infer_images
        # for checkpoint_no in range(checkpoint_start,checkpoint_end,checkpoint_step):
        #     resume = checkpoint_no
        #
        #     os.system("python ../infer.py --experiment_dir ../experiment_dir  --batch_size 32 --resume " + str(
        #         resume) + " --from_txt --src_font ../fonts/source/仓耳今楷03-W03.ttf --src_txt 和 --label " + str(label))
        #     img = Image.open('../experiment_dir/infer/' + str(label) + '/0.png')
        #     img = img.save( images_addr + str(resume) + '.png')

        # save infer_gif
        images = []
        filenames = []

        for i in range(checkpoint_start,checkpoint_end,checkpoint_step):
            img_name = images_addr + str(i) + ".png"

            head = Image.open(img_name)
            headbox = (0, 0, 256, 256)
            head.crop(headbox).save(images_addr + str(i) + "_crop.png")

        for i in range(checkpoint_start,checkpoint_end,checkpoint_step):
            img_name = images_addr + str(i) + "_crop.png"
            filenames.append(img_name)

            if (i == checkpoint_end - checkpoint_step):
                for j in range(1, 40, 1):
                    filenames.append(img_name)

        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(gif_addr + 'transform.gif', images, fps=10)





