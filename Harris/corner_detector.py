import sys
import skimage, skimage.io, pylab, scipy.ndimage.filters
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from operator import itemgetter

class HarrisCornerDetector:
	def __init__(its):
		its.rows = 0
		its.cols = 0

	def load(its, image):
		its.A = skimage.io.imread(image)
		its.A = skimage.img_as_float(its.A)
		A = skimage.color.rgb2gray(its.A)

		its.rows, its.cols = A.shape[0], A.shape[1]
		its.shape = A.shape

		return A

	def gaussian(its, A):
		kernel_5x5 = np.array(
			[[2, 4, 5, 4, 2], 
			[4, 9, 12, 9, 4],
			[5, 12, 15, 12, 5], 
			[4, 9, 12, 9, 4], 
			[2, 4, 5, 4, 2]]) / 159

		# Convolve image using filter
		return scipy.ndimage.filters.convolve(A, kernel_5x5)

	def find_gradients(its, A):
		# Sobel x filter
		sobel_x = np.array(
			[[-1, 0, 1],
			 [-2, 0, 2],
			 [-1, 0, 1]]) 	

		# Sobel y filter
		sobel_y = np.array(
			[[1, 2, 1],
			 [0, 0, 0],
			 [-1, -2, -1]]) 	

		# Gradient Partial
		gx = scipy.ndimage.filters.convolve(A, sobel_x)
		gy = scipy.ndimage.filters.convolve(A, sobel_y)

		return gx, gy

	def find_corners(its, dx, dy, m, t):
		Ix2 = np.array(dx**2)
		Iy2 = np.array(dy**2)
		Ixy = np.array(dx*dy)
		window = (2*m+1)**2		

		corners_list = {}
		e_val = []

		for r in range(its.rows):
			up = max(r-m, 0)
			down = min(r+m+1, its.rows-1) 
			for c in range(its.cols):
				left = max(c-m, 0)
				right = min(c+m+1, its.cols-1)

				ix2 = Ix2[up: down, left: right].sum() / window
				iy2 = Iy2[up: down, left: right].sum() / window
				ixy = Ixy[up: down, left: right].sum() / window

				C = np.array([[ix2, ixy], 
							  [ixy, iy2]]) 

				eigenvalue = np.linalg.det(C) - 0.04*(np.trace(C)**2)
				e_val.append((r, c, eigenvalue))

		max_e = max(e_val, key=lambda x: x[2])[2]
		threshold = t*max_e
		for i in e_val:
			r = i[0]
			c = i[1]
			eigenvalue = i[2]
			if eigenvalue >= threshold:	
				corners_list[(r, c)] = [eigenvalue, "pending"]

		return corners_list

	def reverse_sort(its, corners_list):
		return sorted(corners_list.items(), key=lambda x: x[1][0], reverse=True)
	
	def create_list(its, r, c, m):
		# creates a list of tuple indices used to pop
		up = max(r-m, 0)
		down = min(r+m+1, its.rows-1)
		left = max(c-m, 0)
		right = min(c+m+1, its.cols-1)

		# print(up, down, left, right)
		A = []
		for j in range(up, down+1):
			for i in range(left, right+1):
				A.append((j, i))

		# remove center element
		i = A.index((r,c))
		A.pop(i)

		return A

	def nonmax_suppression(its, L, m):
		# remove corner candidates that are connected to other corner candidates with higher e
		A = dict(its.reverse_sort(L))

		# loop through the list of candidate corners
		for r,c in A:
			val = A[r,c][0]
			status = A[r,c][1]

			if status == "pending":
				A[r,c][1] = True
				# finds neigbors within frame m and returns as a list
				pop_list = its.create_list(r, c, m)

				for i in pop_list:
					if i in A:
						# we have a neighbor and we do not want them
						A[i][1] = False

		# copying "True" values onto final list
		B = {}
		for r,c in A:
			val = A[r,c][0]
			status = A[r,c][1]

			if status is True:
				B[r,c] = val

		return B
		
	def display(its, L):
		plt.imshow(its.A)
		# print("Number of markers:", len(L))

		all_y = [x[0] for x in L.keys()]
		all_x = [y[1] for y in L.keys()]

		plt.scatter(all_x, all_y, marker = '.')
		plt.show()

if __name__ == '__main__':
	image = '../_ref/' + sys.argv[1]
	threshold = float(sys.argv[2])
	window_size = 4

	h = HarrisCornerDetector()
	A = h.load(image)
	smooth_img = h.gaussian(A)
	dx, dy = h.find_gradients(smooth_img)
	l = h.find_corners(dx, dy, window_size, threshold)
	a = h.nonmax_suppression(l, 8)

	h.display(a)