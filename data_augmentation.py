import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import imgaug
from imgaug import augmenters as iaa
import numpy as np
import cv2
import os
import PIL.Image

def compress_to_jpg(image, quality=75):
    quality = quality if quality is not None else 75
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string

def save(fp, image, quality=75):
    image_jpg = compress_to_jpg(image, quality=quality)
    with open(fp, "wb") as f:
        f.write(image_jpg)

seq = iaa.Sequential([
    iaa.Fliplr(1.0), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

image_path = '/home/tombo/workspace/data_augmentation_for_pascal_voc/quokka.jpg'
IMAGES_DIR = '/home/tombo/workspace/data_augmentation_for_pascal_voc/'
image = cv2.imread(image_path)
print(image.dtype)
print(image.ndim)
print(image.shape)

image_aug = seq.augment_images([image])

fp = os.path.join(IMAGES_DIR, "test.jpg")
save(fp,image_aug)

#cv2.imwrite("test.jpg",image_aug)

# with open(fp, "wb") as f:
#     f.write(image_aug)
