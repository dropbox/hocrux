import sys
import random
import PIL.Image

initdir = sys.argv[1]
outdir = sys.argv[2]

def massage_image(filename, output_dir):
    name = filename.split("/")[-1].split(".")[0]
    raw_image = PIL.Image.open(filename)
    raw_image = raw_image.convert("L")
    # the image is 128x128 - crop to center 96x96
    # 128-96 = 32
    # 128-16 = 112
    # this could be zoomed in even more by detecting left/top/right/bottom most non-white pixel
    crop = raw_image.crop((16, 16, 112, 112))
    # crop.save("crop1.bmp", "BMP")
    # resize back to 128x128
    crop = crop.resize((128, 128))
    # crop.save("crop2.bmp", "BMP")
    left_most = 128
    right_most = 0
    top_most = 128
    bottom_most = 0
    for y in range(128):
        for x in range(128):
            if crop.getpixel((y,x)) < 240:
                top_most = min(y, top_most)
                bottom_most = max(y, bottom_most)
                left_most = min(x, left_most)
                right_most = max(x, right_most)

    # print "left, right: ", left_most, " ", right_most
    # print "top, bottom: ", top_most, " ", bottom_most

    for iter in range(10):
        target_img = PIL.Image.new("L", (128, 128), color=255)
        ytrans = random.randint(0, top_most-1 + (128-bottom_most))
        xtrans = random.randint(0, left_most-1 + (128-right_most))
        # print "\t", ytrans, " ", xtrans
        for y in range(top_most, bottom_most+1):
            for x in range(left_most, right_most+1):
                target_img.putpixel((y - top_most + ytrans, x - left_most + xtrans), crop.getpixel((y,x)))
        target_img.save(output_dir + "/" + name + "_" + str(iter) + ".bmp", "BMP")



count = 0
for root, subdirs, files in os.walk(initdir):
    print root
    os.system("mkdir resized/" + root)
    for file in files:
        #print file
        if file.endswith('.bmp'):
            if ++count % 100 == 0:
                print count
            massage_image(root + "/" + file, outdir + root + "/" + file);
            #image = resize_image(root + "/" + file, 32, 32, 'L')
            #scipy.misc.imsave(outdir + root + "/" + file, image)