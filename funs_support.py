"""
SUPPORT FUNCTIONS
"""

import os

def makeifnot(path):
	if not os.path.exists(path):
		print('Making folder: %s' % path)
		os.mkdir(path)
	else:
		print('Folder already exists')

