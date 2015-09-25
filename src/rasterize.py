#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import argparse
import os
import time

import PIL.Image
import numpy as np
import scipy.misc
from google.protobuf import text_format

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2
import enchant

CONFIDENCE_THRESHOLD = 90
WINDOW_WIDTH = 32
WINDOW_HEIGHT = 32
STRIDE_X = 2
STRIDE_Y = 2

OUT_FORMAT = "PNG"
OUT_EXTENSION = ".png"

FULL_WIDTH = -1
FULL_HEIGHT = -1
STRIDE_WIDTH = -1

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    dims = network.input_dim

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file) as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='L'):
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def load_and_splice_image(file_path, height, width, mode='L'):
    global STRIDE_WIDTH
    image = PIL.Image.open(file_path)
    image = image.convert(mode)
    (FULL_WIDTH, FULL_HEIGHT) = image.size

    """black_threshold = 230
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            if image.getpixel((x,y)) > black_threshold:
                image.putpixel((x,y), 255)
            else:
                image.putpixel((x,y), 0)"""

    crops = []
    STRIDE_WIDTH = 0
    print "image.size = " + str(image.size)
    for y in range(0, FULL_HEIGHT - height+1, STRIDE_Y):
        for x in range(0, FULL_WIDTH - width+1, STRIDE_X):
            if y == 0:
                STRIDE_WIDTH += 1
            if len(crops) < 20000:
                crop = image.crop((x, y, x+ width, y + height))
                # crop.save("dump/crop_" + str(y) + "_" + str(x) + OUT_EXTENSION, OUT_FORMAT)
                crop = np.array(crop)
                crops.append(crop)
    print "STRIDE_WIDTH = " + str(STRIDE_WIDTH)
    return crops

def forward_pass(images, net, transformer, batch_size=1):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if scores is None:
            scores = output
        else:
            scores = np.vstack((scores, output))
        # print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))
    print "Processed ", len(caffe_images), " images"

    return scores

def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def getPrintChar(classification):
    # space between words
    if classification == "space":
        return " "
    # this could also be a space between words
    if classification == "blank":
        return " "
    # 0 = noise
    if classification == "underline":
        return " "
    # 1 = space in between characters
    if classification == "overline":
        return " "
    if classification == "between lines":
        return " "
    # space between chars
    if classification == "between chars":
        return " "
    if classification == "part":
        return " "
    if classification == "blank_between":
        return " "
    if classification == "blank_below":
        return " "
    if classification == "blank_above":
        return " "
    return classification


def classify(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    print "target height, width: ", height,  " ", width
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)

    # images = [load_image(image_file, height, width, mode) for image_file in image_files]
    images = load_and_splice_image(image_files[0], height, width, mode)
    labels = read_labels(labels_file)

    # Classify the image
    classify_start_time = time.time()
    scores = forward_pass(images, net, transformer)
    print 'Classification took %s seconds.' % (time.time() - classify_start_time,)

    ### Process the results
    
    indices = (-scores).argsort()[:, :5] # take top 5 results
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i],4)))
        classifications.append(result)

    ''' for index, classification in enumerate(classifications):
        print '{:-^80}'.format(' Prediction for %s ' % str(index))
        for label, confidence in classification:
            print '{:9.4%} - "{}"'.format(confidence/100.0, label)
        print '''

    '''for index, classification in enumerate(classifications):
        print index,
        print '{:9.4%} - "{}"'.format(classification[0][1]/100.0, classification[0][0])
    '''

    # classifcations[index][1..5][label, confidence]
    count = 0
    line = ""
    for index, classification in enumerate(classifications):
        if classification[0][1] > CONFIDENCE_THRESHOLD:
            line += getPrintChar(classification[0][0])
        else:
            line += " "
        if count % STRIDE_WIDTH == 0:
            print line
            line = ""
        count += 1
    return classifications

isMarked = []
def generateConnectedComponent(chars, y, x):
    global isMarked
    # print "y, x", y, " ", x
    if y < 0 or x < 0 or y >= len(chars) or x >= len(chars[y]):
        return []
    if isMarked[y][x]:
        return []
    if chars[y][x] == " ":
        return []
    isMarked[y][x] = True
    r = [(y,x)]
    r += generateConnectedComponent(chars, y-1, x-1)
    r += generateConnectedComponent(chars, y-1, x)
    r += generateConnectedComponent(chars, y-1, x+1)
    r += generateConnectedComponent(chars, y, x-1)
    r += generateConnectedComponent(chars, y, x+1)
    r += generateConnectedComponent(chars, y+1, x-1)
    r += generateConnectedComponent(chars, y+1, x)
    r += generateConnectedComponent(chars, y+1, x+1)
    return r
    

def predictLine(classifications):
    ''' returns False if line skipped (whitespace etc) '''
    ''' if True is returned, skipped the next 16 lines '''
    dchars = []
    for y in range(len(classifications)):
        line = ""
        for c in classifications[y]:
            if c[0][1] > CONFIDENCE_THRESHOLD:
                line += getPrintChar(c[0][0])
            else:
                line += " "
        # print "adding line: ", line
        dchars.append(line)
        
    if dchars[0].count(" ") >= len(dchars[0])-2:
        print "skipping this line"
        return None

    # count the biggest connected components 
    charLine = []
    global isMarked
    isMarked = [[False] * len(line) for line in dchars]
    for y in range(len(dchars)):
        for x in range(len(dchars[y])):
            if dchars[y][x] != " " and not isMarked[y][x]:
                indices = generateConnectedComponent(dchars, y, x)
                # print "y, x, char, size: ", y, " ", x, " ", dchars[y][x], " ", len(indices)
                # take max char
                charCount = {}
                for (py, px) in indices:
                    if not dchars[py][px] in charCount:
                        charCount[dchars[py][px]] = 0
                    charCount[dchars[py][px]] += 1
                maxCount = 0
                componentChar = ""
                for k in charCount:
                    if charCount[k] > maxCount:
                        maxCount = charCount[k]
                        componentChar = k
                # print "\t component char: ", componentChar
                charLine.append((x, componentChar, maxCount))

    charLine = sorted(charLine, key=lambda comp: -comp[2])
    # grab the biggest components by size, don't add if x-coordinate is too close
    # print "charLine: ", charLine
    bigLine = []
    for (x, ch, sz) in charLine:
        tooClose = False
        for (x2, ch2) in bigLine:
            # consider adjusting this parameter
            if abs(x-x2) < 5:
                tooClose = True
                break
        if not tooClose:
            bigLine.append((x, ch))
    bigLine = sorted(bigLine, key=lambda comp: comp[0])
    print "bigLine: ", bigLine    
    return bigLine

def containsWordSeparator(x1, x2, classifications):
    ''' returns true if "space" or "blank" occurs anywhere between x1 and x2 '''
    for pxLine in classifications:
        for x in range(x1+1, x2):
            if pxLine[x][0][1] > CONFIDENCE_THRESHOLD and \
                    (pxLine[x][0][0] == "space" or pxLine[x][0][0] == "blank"):
                return True
    return False
    
def makeWords(bigLine, classifications):
    words = []
    currentWord = ""
    for index in range(len(bigLine)):
        # make space?
        # check if big space (word separator) occurs between x1 and x2
        if index > 1 and bigLine[index][0] - bigLine[index-1][0] > 6 and \
                containsWordSeparator(bigLine[index-1][0], bigLine[index][0], classifications):
            # there's a space here, dump current word
            if currentWord != "":
                words.append(currentWord)
                currentWord = ""

        currentWord += bigLine[index][1]

    if currentWord != "":
        words.append(currentWord)
    return words


# old predictLine
''' def predictLine(classifications):
    assert(len(classifications) == STRIDE_WIDTH)
    allChars = [x[0][0] for x in classifications]
    # print allChars
    chars = [x[0][0] for x in classifications if x[0][1] > CONFIDENCE_THRESHOLD]
    # print chars
    # filter out 0's (noise)
    # treat 1's as spaces
    chars = filter(lambda c: c != '0', chars)
    # transform all same-letter-pairs (XX) into (X)
    for i in range(len(chars)):
        if len(chars[i]) > 1:
            chars[i] = chars[i][0]
    # 1/4 the number of repeated characters, rounding down, at least 1
    singleChars = []
    index = 0
    while index < len(chars):
        start = index
        while index < len(chars) and chars[index] == chars[start]:
            index += 1
        singleChars += [chars[start]] * max(1, int((index-start+1)/4.0))
    # split words by spaces
    words = []
    currentWord = ""
    for c in singleChars:
        if c == "1" and currentWord != "":
            words.append(currentWord)
            currentWord = ""
        elif c != "1":
            currentWord += c
    if currentWord != "":
        words.append(currentWord)
    print "words: ", words
    # check against en_US spelling
    endict = enchant.request_dict("en_US")
    dictWords = []
    for w in words:
        if endict.check(w):
            dictWords.append(w)
        else:
            suggestions = endict.suggest(w)
            # just append the first one
            if len(suggestions) > 0:
                dictWords.append(suggestions[0])
            # else we have unrecoverable garbage?
    print "\t dict: ", dictWords
'''

def runThruDictionary(words):
    # check against en_US spelling
    endict = enchant.request_dict("en_US")
    dictWords = []
    for w in words:
        if endict.check(w):
            dictWords.append(w)
        else:
            suggestions = endict.suggest(w)
            # just append the first one
            if len(suggestions) > 0:
                dictWords.append(suggestions[0])
            # else we have unrecoverable garbage?                
    return dictWords


def predictWords(classifications):
    # STRIDE_WIDTH is one line
    lineIndex = 0
    allLines = []
    while (lineIndex + (WINDOW_HEIGHT/STRIDE_Y) - 1) * STRIDE_WIDTH < len(classifications):
        print "#", lineIndex, ": ",
        # splice here
        r = predictLine([classifications[(lineIndex + i) * STRIDE_WIDTH : (lineIndex + i +1) * STRIDE_WIDTH] 
                         for i in range(0, WINDOW_HEIGHT/STRIDE_Y)])
        if not r:
            lineIndex += 1
            continue
        
        words = makeWords(r,
                          [classifications[(lineIndex + i) * STRIDE_WIDTH : (lineIndex + i +1) * STRIDE_WIDTH] 
                           for i in range(0, WINDOW_HEIGHT/STRIDE_Y)])
        print "words: ", words
        dictWords = runThruDictionary(words)
        print "\t: ", dictWords
        lineIndex += WINDOW_HEIGHT/STRIDE_Y
        allLines.append(dictWords)
    return allLines
    

def writeTextToFile(imgfilename, lines):
    txtname = imgfilename.split(".")[0] + ".txt"
    count = 0
    while os.path.isfile(txtname):
        count += 1
        txtname = imgfilename.split(".")[0] + "." + str(count) + ".txt"
            
    f = open(txtname, "w")
    for line in lines:
        for w in line:
            f.write(w + " ")
        f.write("\n")
    f.close()
    

if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Classification example - DIGITS')

    ### Positional arguments

    parser.add_argument('caffemodel',   help='Path to a .caffemodel')
    parser.add_argument('deploy_file',  help='Path to the deploy file')
    parser.add_argument('image',        help='Path to an image')

    ### Optional arguments

    parser.add_argument('-m', '--mean',
            help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels',
            help='Path to a labels file')
    parser.add_argument('--nogpu',
            action='store_true',
            help="Don't use the GPU")

    args = vars(parser.parse_args())

    image_files = [args['image']]

    classifications = classify(args['caffemodel'], args['deploy_file'], image_files,
                               args['mean'], args['labels'], not args['nogpu'])

    lines = predictWords(classifications)

    writeTextToFile(image_files[0], lines)

    print 'Script took %s seconds.' % (time.time() - script_start_time,)

