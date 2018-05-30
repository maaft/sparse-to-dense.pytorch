import sys
from os import walk
import os
import numpy as np
from scipy.ndimage import imread

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(raw_input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

path = sys.argv[1]

files = {}

for (dirpath, dirnames, filenames) in walk(path):
    for file in filenames:
        if ".png" in file:
            img = imread(os.path.join(dirpath, file))
            if img.ndim == 3:
                img = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

            files[os.path.join(dirpath, file)] = np.histogram(img, 256, normed=True)[0]

duplicates = []

for name1, hist1 in files.iteritems():
    for name2, hist2 in files.iteritems():
        if name1 == name2:
            continue
        dist = np.linalg.norm(hist1 - hist2)
        if dist < 0.01:
            duplicates.append((name1, name2))

for duplicate in duplicates:
    print("Found duplicates '{}' and '{}'!".format(duplicate[0], duplicate[1]))
    if yes_or_no("Remove first?"):
        os.remove(duplicate[0])
    else:
        continue