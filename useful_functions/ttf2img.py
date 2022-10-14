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

def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]

def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
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
    return img

def draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # src_img.show()
    # convert to gray img
    example_img = example_img.convert('L')
    # example_img.show()
    return example_img


def font2font(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    # if filter_by_hash:
    #     filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
    #     print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_font2font_example(c, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


def get_ground_truth_img(is_combined_img,src_font_addr,dst_font_addr,label,ch, sample_dir):

    char_size = 256
    canvas_size = 256
    x_offset = 0
    y_offset = 0
    filter = True

    src_font = ImageFont.truetype(src_font_addr, size=char_size)
    dst_font = ImageFont.truetype(dst_font_addr, size=char_size)

    if(is_combined_img == 1):
        # method2 --- combine character
        filter_hashes = set()
        dst_img = draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
    else:
        # method1 --- single character
        dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)

    dst_img.save(os.path.join(sample_dir, "ground_truth.jpg"))


if __name__ == "__main__":

    src_font_addr = '../fonts/source/仓耳今楷03-W03.ttf'
    dst_font_addr = '../fonts/target/蔡云汉简体行书书法字体.ttf'

    label = 0
    ch = '锋'

    sample_dir = './'
    sample_count = 1

    char_size = 256
    canvas_size = 256
    x_offset = 0
    y_offset = 0
    filter = True

    src_font = ImageFont.truetype(src_font_addr, size=char_size)
    dst_font = ImageFont.truetype(dst_font_addr, size=char_size)

    # method1 --- single character
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)

    src_img.show()
    dst_img.show()


    # # method2 --- combine character
    # # filter_hashes = set()
    # # dst_img = draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
    # src_img.save(os.path.join(sample_dir, "source.jpg" ))
    # # dst_img.save(os.path.join(sample_dir, "ground_truth.jpg" ))
