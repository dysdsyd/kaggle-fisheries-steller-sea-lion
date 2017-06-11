# https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/33900
# https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/34546

#import numpy as np
#import pandas as pd
#import cv2
#import sys
import os
#import matplotlib.pyplot as plt

import PIL 
import PIL.Image

tgtsize = (512, 512)

#from multiprocessing import Pool

def convimage(tup):
    infile, outfile = tup

    img = PIL.Image.open(infile)

    img2 = img.resize(tgtsize, PIL.Image.BICUBIC)
    img2.save(outfile)

def procdir(indir, outdir):
    try:
        os.mkdir(outdir)
    except:
        pass

    flist = []

    for f in os.listdir(indir):
        if f[-3:] != 'jpg':
            print(2)
            continue
        #print(f)
        flist.append((indir + f, outdir + f.split('.')[0] + '.png'))
    return flist
#==============================================================================
#     with Pool(2) as p:
#         p.map(convimage, flist)
#==============================================================================

indir = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/Test/'
outdir = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/test_images_{0}x{1}/'.format(*tgtsize)
flist = procdir(indir, outdir)
for tup in flist:
    convimage(tup)
    
indir = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/Train/'
outdir = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/train_images_{0}x{1}/'.format(*tgtsize)
flist = procdir(indir, outdir)
for tup in flist:
    convimage(tup)