from PIL import Image
import numpy as np

#img1 = Image.open('data/synthia/RGB/0003440.png')
img2 = Image.open('data/synthia/RGB/0000617.png')
img3 = Image.open('data/synthia/RGB/0001182.png')
img4 = Image.open('data/synthia/RGB/0003310.png')

#img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
img4 = np.array(img4)

img_list = [img2, img3, img4] #[img1, img2, img3, img4]

for img in img_list:
    print(type(img))
    print(img.shape)
# print(type(img1), type(img2), type(img3), type(img4))
# print(img1.shape, img2.shape, img3.shape, img4.shape)


