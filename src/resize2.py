import sys
import PIL.Image

if __name__ == "__main__":
    # resize the target image to (at least) 128x128 box
    filename = sys.argv[1]
    print "Converting " + filename
    image = PIL.Image.open(filename)
    # image.size = (width, height)
    (w, h) = image.size
    ratio = max(128.0/w, 128.0/h)
    image = image.resize((int(w * ratio), int(h * ratio)))
    image.save(filename)
