"""
SUPPORT FUNCTIONS
"""

import os
import socket
import sys

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
		dir_GI = os.path.join(dir_base, '..')
	else:
		sys.exit('Where are we?!')
	return dir_GI


def makeifnot(path):
	if not os.path.exists(path):
		print('Making folder: %s' % path)
		os.mkdir(path)
	else:
		print('Folder already exists')

