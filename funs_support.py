"""
SUPPORT FUNCTIONS
"""

import os
import socket
import sys
import numpy as np

def stopifnot(arg, msg):
    if not arg:
        sys.exit(msg)

def no_diff(x, y):
	uu = np.union1d(x, y)
	ii = np.intersect1d(x, y)
	check = len(np.setdiff1d(uu, ii)) == 0
	return check

def listfiles(path):
	return sorted(os.listdir(path))

def listfolds(path):
	return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def find_dir_GI():
	dir_base = os.getcwd()
	cpu = socket.gethostname()
	# Set directory based on CPU name
	if cpu == 'RT5362WL-GGB':
		print('On predator machine')
		if os.name == 'nt':
			print('Using windows')
			dir_GI = 'D:\\projects\\GIOrdinal'
		elif os.name=='posix':
			print('Using WSL')
			dir_GI = '/mnt/d/projects/GIOrdinal'
		else:
			print('Not sure which operating system?!')
	elif cpu == 'snowqueen':
		print('On snowqueen machine')
		dir_GI = '/data/GIOrdinal'
	else:
		sys.exit('Where are we?!')
	return dir_GI


def makeifnot(path):
	if not os.path.exists(path):
		print('Making folder: %s' % path)
		os.mkdir(path)
	else:
		print('Folder already exists')

def random_crop(img, height, width, crop_size, ss):
    np.random.seed(ss)
    yidx = np.random.choice(np.arange(height - crop_size))
    xidx = np.random.choice(np.arange(width - crop_size))
    cropped = img[yidx:(yidx + crop_size + 1), xidx:(xidx + crop_size) + 1].copy()
    return cropped, yidx, xidx
