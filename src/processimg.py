import sys
import PIL.Image
import PIL.ImageEnhance

if __name__ == "__main__":
    img = PIL.Image.open(sys.argv[1])
    img = img.convert("L")
    img = img.resize((img.size[0]/7, img.size[1]/7))
    en = PIL.ImageEnhance.Contrast(img)
    img2 = en.enhance(2.5)
    br = PIL.ImageEnhance.Brightness(img2)
    img2 = br.enhance(3.0)
    ''' for y in range(img.size[0]):
        for x in range(img.size[1]):
            if img.getpixel((y,x)) < 130:
                img.putpixel((y,x), 0)
            else:
                img.putpixel((y,x), 255)
    '''
    img2.save("output.jpg")





