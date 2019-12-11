from __future__ import division
import numpy as np
import math
import bcfstore
import util2
import os
from PIL import Image
from StringIO import StringIO
affine2d = __import__('affine2d')

def initialize(data_dir, test):
    """ data_dir: directory of samples
        test: directory of labels

        used to initialize the datastore and labelstore constructs of BCF
    """
    dataStore, labelStore = util2.read_bcf_file(data_dir,test)
    return dataStore, labelStore

tp_dataStore, tp_labelStore = initialize('./',0)

fonttxt = open("./fontlist.txt", "r")
fonttxt = fonttxt.read().split()
fontlist = []
for font in fonttxt:
    fontlist.append(font)
count = 0

for i in range(1567000, tp_dataStore.size()):
    im = Image.open(StringIO(tp_dataStore.get(i)))

    # get fontlist and for each index in label, add count to end of name
    index = tp_labelStore[i]
    if i > 0 and index == tp_labelStore[i-1]:
        count += 1
    else:
        count = 0

    fontname = fontlist[index]
    fontfile = fontname + str(count)
    
    savepath = "syn_train/" + fontname
    # Make directory if it does not exist.
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    im.save(savepath + "/" + fontfile + ".png", "PNG")
