import sys
import os
import random
import PIL.Image

BLACK_THRESHOLD = 240
WIDTH = 128
HEIGHT = 128

def get_black_rectangle(img):
    left_most = WIDTH
    right_most = 0
    top_most = HEIGHT
    bottom_most = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if img.getpixel((x,y)) < BLACK_THRESHOLD:
                top_most = min(y, top_most)
                bottom_most = max(y, bottom_most)
                left_most = min(x, left_most)
                right_most = max(x, right_most)
    return (top_most, bottom_most, left_most, right_most)


# assumes both images are HEIGHTxWIDTH
# set rescale_to_32 to rescale; stays at 128 by default
def make_char_space(char1, char2, output_file, rescale_to_32):
    A = PIL.Image.open(char1)
    B = PIL.Image.open(char2)
    A = A.convert("L")
    B = B.convert("L")
    
    output = PIL.Image.new("L", (HEIGHT, WIDTH), color=255)
    # blackX = (y1, y2, x1, x2)
    blackA = get_black_rectangle(A)
    print "A rect: ", blackA
    blackB = get_black_rectangle(B)
    
    # copy right side of blackA(at least 5 px; 1/3 of char width, at most 32 px, no more than all of blackA) to the left hand side of output
    ncolumns = random.randint(min(5, (blackA[3] - blackA[2])/3), min(32, blackA[3] - blackA[2]))
    print "left: ", ncolumns, " columns"
    for y in range(blackA[0], blackA[1]+1):
        for x in range(ncolumns):
            output.putpixel((x, y), A.getpixel((blackA[3] - ncolumns + x, y)))

    # copy left side of blackB (at least 1 px, at most 32 px, no more than all of blackB) to the right hand side of output 
    ncolumns = random.randint(min(5, (blackB[3] - blackB[2])/3), min(32, blackB[3] - blackB[2]))
    print "right: ", ncolumns, " columns"
    for y in range(blackB[0], blackB[1]+1):
        for x in range(ncolumns):
            output.putpixel((WIDTH - ncolumns + x, y), B.getpixel((blackB[2] + x, y)))
    
    if rescale_to_32:
        output = output.resize((32,32))

    # the space gap is at least 64 pixels wide
    ext = "png" #char1.split(".")[-1]
    output.save(output_file + "." + ext, ext)    

if __name__ == "__main__":
    char1 = sys.argv[1]
    char2 = sys.argv[2]
    make_char_space(char1, char2, "output", True)
