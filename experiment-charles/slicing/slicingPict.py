#train
#format 1  5616 * 3744 
#format 2  4992 * 3328
#2 exception 3744 * 5616  (530 & 638)

#test
#format   5760 * 3840  
#format   5616 * 3744
#format   5184 * 3456
#format   4608 * 3456
#1 exception in test: 881 * 1280

im = Image.open("F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/slicetest/0_13_07.png")
im =im.resize(size)
im.save("F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/slicetest/0_13_07.jpeg", "JPEG")





import image_slicer
image_slicer.slice('F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/slicetest/0.jpg', 225)

import os, sys
from PIL import Image

size = 244, 224




im = Image.open("F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/slicetest/0_13_07.png")
im =im.resize(size)
im.save("F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/slicetest/0_13_07.jpeg", "JPEG")



