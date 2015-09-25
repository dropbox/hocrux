import sys
import os
import random
import PIL.Image
import string

initdir = sys.argv[1]
outdir = sys.argv[2]

max_count = -1
left_per = 3
right_per = 3
left_right_per = 3
self_per = 2
noise = 500
crop_threshold = 240
base_count = 1000

min_scale = 0.35
max_scale = 0.4

in_width = 128
in_height = 128
in_extension = ".bmp"

out_width = 32
out_height = 32
out_format = "PNG"
out_extension = ".png"

all_files = []
saved_counts = {}
cropped_image_cache = {}

if not os.path.exists(outdir):
    os.system("mkdir " + outdir)

"""for x in string.ascii_lowercase:
    os.system("mkdir " + outdir + "/" + x)
os.system("mkdir " + outdir + "/unknown")
os.system("mkdir " + outdir + "/0")
os.system("mkdir " + outdir + "/1")"""

def get_file_char(file):
    char = ""
    for x in string.ascii_lowercase:
        if "_"+x+"_" in file:
            char = x
            break
    for x in string.ascii_uppercase:
        if "_"+x+"_" in file:
            char = x.lower() #x + "2"
            break
    return char

count = 0
for root, subdirs, files in os.walk(initdir):
    if max_count != -1 and count >= max_count:
            break
    #print root
    for file in files:
        char = get_file_char(file)
        if file.endswith(in_extension) and char != "":
            all_files.append(root + "/" + file)
        count += 1;
        if max_count != -1 and count >= max_count:
            break

print "all files accounted for"


def get_image_crop(image):
    (w, h) = image.size
    l = w
    r = 0
    t = h
    b = 0
    pixels = image.load()
    for x in range(w):
        for y in range(h):
            if pixels[x,y] < crop_threshold:
                l = min(x, l)
                t = min(y, t)
                r = max(x, r)
                b = max(y, b)
    return [l,t,r+1,b+1]

def get_cropped_image(file):
    if file in cropped_image_cache:
        return cropped_image_cache[file]
    raw_image = PIL.Image.open(file)
    raw_image = raw_image.convert("L")
    (l, t, r, b) = get_image_crop(raw_image)
    result = raw_image.crop((l,t,r,b))
    cropped_image_cache[file] = result
    return result

def get_scaled_image(file):
    scale = min_scale + (max_scale - min_scale) * random.random()
    image = get_cropped_image(file)
    (w, h) = image.size
    scale = min(scale, 0.8 * out_width / w, 0.8 * out_height / h)
    w = int(w * scale)
    h = int(h * scale)
    image = image.resize((w, h), PIL.Image.ANTIALIAS)
    return image

def add_noise(image):
    pixels = image.load()
    for i in range(noise):
        x = int(random.random() * out_width)
        y = int(random.random() * out_height)
        pixel = pixels[x,y]
        if pixel > 128:
            pixel -= 10
            pixels[x,y] = pixel
        else:
            pixel += 10
            pixels[x,y] = pixel


def save_image(dir, char, image):
    if not char in saved_counts:
        saved_counts[char] = 0
    char_count = saved_counts[char]
    char_count = char_count + 1
    saved_counts[char] = char_count
    dir_path = dir + "/" + char
    if not os.path.exists(dir_path):
        os.system("mkdir " + dir_path)
    path = dir_path + "/" + "img_" + char + "_" + str(char_count) + out_extension
    image.save(path, out_format)

def scale_to_fit(image, width, height):
    (w,h) = image.size
    scale = min(float(width) / float(w), float(height) / float(h))
    w = int(scale * w)
    h = int(scale * h)
    return image.resize((w, h), PIL.Image.ANTIALIAS)

def make_training(path, add_left, add_right):
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    center_image = get_scaled_image(path)
    (w,h) = center_image.size
    center_left = int((out_width - w) * (0.4 + 0.2 * random.random()))
    center_top = int((out_height - h) * (0.4 + 0.2 * random.random()))
    center_right = center_left + w
    center_bottom = center_top + h

    image.paste(center_image, (center_left, center_top))

    if add_left:
        left_image = get_scaled_image(random.choice(all_files))
        (w,h) = left_image.size
        r = center_left - random.randrange(0,4)
        b = center_bottom
        t = b - h
        l = r - w
        image.paste(left_image, (l, t))

    if add_right:
        right_image = get_scaled_image(random.choice(all_files))
        (w,h) = right_image.size
        l = center_right + random.randrange(0,4)
        b = center_bottom
        t = b - h
        r = l + w
        image.paste(right_image, (l, t))

    add_noise(image)

    char = "unknown"
    for x in string.ascii_lowercase:
        if "_"+x+"_" in path:
            char = x
            break
    for x in string.ascii_uppercase:
        if "_"+x+"_" in path:
            char = x.lower() #x + "2"
            break

    save_image(outdir, char, image)


def make_between_chars():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    x1 = int(out_width * (0.4 + 0.2 * random.random()))

    path1 = random.choice(all_files)
    image1 = get_scaled_image(path1)
    (w,h) = image1.size
    r = x1 - random.randrange(0,3)
    t = int((out_height - h) * (0.4 + 0.2 * random.random()))
    b = t + h
    l = r - w
    if l > 0:
        return False

    image.paste(image1, (l, t))

    path2 = random.choice(all_files)
    image2 = get_scaled_image(path2)
    (w,h) = image2.size
    t = b - h
    l = x1 + random.randrange(0,3)
    r = l + w
    if r < out_width:
        return False

    image.paste(image2, (l, t))

    add_noise(image)

    save_image(outdir, "between_chars", image)
    return True


def make_char_space():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    x1 = int(out_width * (0.4 + 0.2 * random.random()))
    gap = int(out_width * (0.4 + 0.2 * random.random()))

    path1 = random.choice(all_files)
    image1 = get_scaled_image(path1)
    (w,h) = image1.size
    r = x1 - gap / 2
    t = int((out_height - h) * (0.4 + 0.2 * random.random()))
    b = t + h
    l = r - w
    if l > 0:
        return False

    image.paste(image1, (l, t))

    path2 = random.choice(all_files)
    image2 = get_scaled_image(path2)
    (w,h) = image2.size
    t = b - h
    l = x1 + gap / 2
    r = l + w
    if r < out_width:
        return False

    image.paste(image2, (l, t))

    add_noise(image)

    save_image(outdir, "space", image)
    return True

def make_blank_space():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)
    add_noise(image)
    save_image(outdir, "blank", image)


def make_between_lines():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    y1 = int((0.3 + 0.4 * random.random()) * out_height)
    max_gap = min(out_height - y1, y1)
    gap = int(max_gap * (0.6 + 0.2 * random.random()))

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    l = int(random.random() * (out_width - w))
    t = y1 - h - gap / 2
    if t > 0:
        return False
    image.paste(img, (l, t))

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    l = int(random.random() * (out_width - w))
    t = y1 + gap / 2
    b = t + h
    if b < out_height:
        return False
    image.paste(img, (l, t))

    add_noise(image)
    save_image(outdir, "between_lines", image)
    return True


def make_char_too_high():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    b = y1
    t = b - h
    l = int(random.random() * (out_width - w))
    image.paste(img, (l, t))

    add_noise(image)
    save_image(outdir, "underline", image)


def make_char_too_high2():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    x1 = int((0.1 + 0.9 * random.random()) * out_width)
    gap = int(0.2 * random.random() * out_width)

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    b = y1
    t = b - h
    l = x1 - w - gap / 2
    image.paste(img, (l, t))

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    b = y1
    t = b - h
    l = x1 + gap / 2
    image.paste(img, (l, t))

    add_noise(image)
    save_image(outdir, "underline", image)


def make_char_too_low():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(out_height - min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    l = int(random.random() * (out_width - w))
    t = y1
    image.paste(img, (l, t))

    add_noise(image)
    save_image(outdir, "overline", image)


def make_char_too_low2():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    x1 = int((0.1 + 0.9 * random.random()) * out_width)
    gap = int(0.2 * random.random() * out_width)

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(out_height - min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    l = x1 - w - gap / 2
    t = y1
    image.paste(img, (l, t))

    img = get_scaled_image(random.choice(all_files))
    (w,h) = img.size
    y1 = int(out_height - min(h, out_height / 2) * (0.2 + 0.8 * random.random()))
    l = x1 + gap / 2
    t = y1
    image.paste(img, (l, t))

    add_noise(image)
    save_image(outdir, "overline", image)


def make_char_pair():
    image = PIL.Image.new("L", (out_width, out_height), 0xff)

    x1 = int(out_width * (0.4 + 0.2 * random.random()))

    path1 = random.choice(all_files)
    image1 = get_scaled_image(path1)
    (w,h) = image1.size
    if w > x1:
        return False
    r = x1
    t = int((out_height - h) * (0.4 + 0.2 * random.random()))
    b = t + h
    l = r - w
    image.paste(image1, (l, t))

    path2 = random.choice(all_files)
    image2 = get_scaled_image(path2)
    (w,h) = image2.size
    if w > out_width - x1:
        return False
    b = t + h
    t = b - h
    l = x1
    r = l + w
    image.paste(image2, (l, t))

    add_noise(image)

    char1 = get_file_char(path1)
    char2 = get_file_char(path2)
    save_image(outdir, char1 + char2, image)
    return True


def make_training_set(file):
    for i in range(left_per):
        make_training(file, True, False)
    for i in range(right_per):
        make_training(file, False, True)
    for i in range(left_right_per):
        make_training(file, True, True)
    for i in range(self_per):
        make_training(file, False, False)

for i in range(10 * base_count):
    make_char_space()
print "char spaces complete"

for i in range(20 * base_count):
    make_between_chars()
print "between chars complete"

for i in range(10 * base_count):
    make_char_too_high()
    make_char_too_high2()
print "char too high complete"

for i in range(10 * base_count):
    make_char_too_low()
    make_char_too_low2()
print "char too low complete"

for i in range(10 * base_count):
    make_between_lines()
print "char too high and low complete"

for i in range(10 * base_count):
    make_blank_space()
print "blank space complete"

count = 0
for file in all_files:
    make_training_set(file)
    count = count + 1
    if count % 100 == 0:
        print count