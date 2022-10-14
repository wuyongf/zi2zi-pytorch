import os

from fontTools.ttLib import TTFont
import numpy as np

def get_char_list_from_ttf(font_file):
    ' 给定font_file,获取它的中文字符 '
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()

    unicode_list = []
    for key, _ in m_dict.items():
        # 中日韩统一表意文字 范围: 4E00—9FFF // CJK Unified Ideographs. Range: 4E00—9FFF
        if key >= 0x4E00 and key <= 0x9FFF:
            unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

# font_file_addr = '../fonts/target/'
font_file_addr = '../fonts/source/'
font_file_name = '仓耳今楷03-W03'
font_file_format = ".ttf"

font_file = font_file_addr + font_file_name + font_file_format

chars = get_char_list_from_ttf(font_file)
print(chars)

npchars_unsort1 = np.array(chars)
npchars = npchars_unsort1[None, :]

print(npchars.shape)
shape = npchars.shape[1]
# print(npchars[0][1]) # for index

# Write .txt files
npchars_addr = "../fonts/target_exist_char/"
os.makedirs(npchars_addr, exist_ok=True)

np.savetxt(npchars_addr + '已有字_' + str(shape) + '_' +font_file_name + '.txt', npchars, delimiter="", fmt="%s", encoding='utf-8')
