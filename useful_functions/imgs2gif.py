from fontTools.ttLib import TTFont
import imageio
from PIL import Image, ImageFont, ImageDraw

if __name__ == "__main__":

    images = []
    filenames = []

    drive_addr = './result/0_蔡云汉简体行书书法字体/infer_images/'

    for i in range(500, 25000, 500):
        img_name = drive_addr + str(i) + ".png"

        head = Image.open(img_name)
        headbox = (0,0,256,256)
        head.crop(headbox).save( drive_addr + str(i) + "_crop.png")

    for i in range(500, 25000, 500):
        img_name = drive_addr + str(i) + "_crop.png"
        filenames.append(img_name)

        if(i == 25000 -500):
            for j in range (1,40,1):
                filenames.append(img_name)

    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('result/0_蔡云汉简体行书书法字体/infer_gif/transform.gif', images, fps=10)
