import os
import sys

import argparse
import numpy as np

from PIL import Image, ImageFont, ImageDraw
import json
import collections
import re
from fontTools.ttLib import TTFont
from tqdm import tqdm
import random

from torch import nn
from torchvision import transforms

from utils.charset_util import processGlyphNames



if __name__ == "__main__":


    # image = Image.new("RGB", [320, 320])
    # draw = ImageDraw.Draw(image)
    # a = u"ひらがな - Hiragana, 히라가나"
    # font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 14)
    # draw.text((50, 50), a, font=font)
    # image.save("a.png")

    sample_dir = ''
    sample_count = 1000

    char_size = 256
    canvas_size = 256
    x_offset = 0
    y = offset = 0

    ch = u"泽"

    dst = '../fonts/source/仓耳今楷03-W03.ttf'
    # dst = '../fonts/target/草檀斋毛泽东字体.ttf'
    dst_font = ImageFont.truetype(dst, size=char_size, encoding="utf-8")

    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=dst_font)
        img.show()
    except OSError: None
    bbox = img.getbbox()
    if bbox is None: None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d: None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError: None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    # img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    img = img.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)




    # body = Image.open("红薯夫妇-1.jpg")
    #
    # head = Image.open("红薯夫妇-10.jpg")
    # headbox = (0,0,256,256)
    #
    # head.crop(headbox).save("head.jpg")
    # head_crop = Image.open("./head.jpg")
    #
    # body.paste(head_crop, (0,0))
    # body.save("out.jpg")