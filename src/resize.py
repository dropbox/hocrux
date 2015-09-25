import sys
import os
import PIL.Image
import scipy.misc
import numpy as np

initdir = sys.argv[1]

def resize_image(path, height, width, mode='L'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

os.system("mkdir resized")

print "resizing " + initdir
count = 0
for root, subdirs, files in os.walk(initdir):
    print root
    os.system("mkdir resized/" + root)
    for file in files:
        #print file
        if file.endswith('.bmp'):
            if count % 100 == 0:
                print count
            image = resize_image(root + "/" + file, 48, 48, 'L')
            scipy.misc.imsave("resized/" + root + "/" + file, image)
            ++count


