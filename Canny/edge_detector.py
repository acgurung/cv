import sys
import skimage, skimage.io, pylab, scipy.ndimage.filters
import numpy as np
from math import pi

class CannyEdgeDetector:
	def __init__(its):
		its.rows = 0 
		its.cols = 0

	def load(its, image):
		# Load image
		A = skimage.io.imread(image) 

		# Convert image to float64
		A = skimage.img_as_float(A)

		# Extract luminance as 2D array
		A = skimage.color.rgb2gray(A)

		its.rows = A.shape[0]
		its.cols = A.shape[1]
		its.shape = A.shape
		its.visited = np.full(its.shape, False)
		return A

	def gaussian(its, A):
		# Gaussian filter
		kernel_3x3 = np.array(
			[[1, 2, 1],
			[2, 4, 2],
			[1, 2, 1]]) / 16

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

		edge_strength = its.gradient_magnitude(gx, gy)
		edge_orientation = its.arctan(gx, gy)

		return edge_strength, edge_orientation

	def gradient_magnitude(its, dx, dy):
		# Edge strength (magnitude of the gradient)
		return np.sqrt(dx**2 + dy**2)

	def arctan(its, dx, dy):
		# Edge orientation (degrees)
		return (np.arctan2(dx, dy) / pi) * 180 # Arctan2 returns values between -pi, pi

	def nonmax_suppression(its, F, D):
		# Find angle closest to the orientation of the pixel
		theta = np.zeros(D.shape) 
		for r in range(its.rows):
			for c in range(its.cols):
				elem = D[r, c]

				if (elem > -22.5 and elem < 22.5) or (elem < -157.5 and elem > 157.5):
					theta[r, c] = 0 
				if (elem > 22.5 and elem < 67.5) or (elem > -157.5 and elem < -112.5):
					theta[r, c] = 45 
				if (elem > 67.5 and elem < 112.5) or (elem > -112.5 and elem < -67.5):
					theta[r, c] = 90 
				if (elem > 112.5 and elem < 157.5) or (elem > -67.5 and elem < -22.5):
					theta[r, c] = 135 

		# If edge strength F[r, c] < its neighbors along the angle, then I[r, c] = 0
		# otherwise, I[r, c] = F[r, c]. 		
		I = np.copy(F)
		for r in range(its.rows-1):
			for c in range(its.cols-1):
				# N and S
				if theta[r, c] == 0 and (F[r+1, c] > F[r, c] or F[r-1, c] > F[r, c]):
					I[r, c] = 0
				# NE and SW
				if theta[r, c] == 45 and (F[r-1, c+1] > F[r, c] or F[r+1, c-1] > F[r, c]):
					I[r, c] = 0
				# W and E
				if theta[r, c] == 90 and (F[r, c+1] > F[r, c] or F[r, c-1] > F[r, c]):
					I[r, c] = 0
				# NW and SE
				if theta[r, c] == 135 and (F[r-1, c-1] > F[r, c] or F[r+1, c+1] > F[r, c]):
					I[r, c] = 0
		return I

	def hysteresis_thresholding(its, A, high, low):
		# Prevents passing A to recursive call
		its.A = np.copy(A)
		
		# Label pixels 
		for r in range(its.rows-1):
			for c in range(its.cols-1):
				#Not edge
				if (its.A[r, c] < low):
					its.A[r, c] = 0	
				#Strong edge
				elif (its.A[r, c] > high):
					its.A[r, c] = 1	
				#Weak edge
				else:
					its.A[r, c] = 0.5

		# Create a path among edges
		for r in range(its.rows-1):
			for c in range(its.cols-1):
				# if we're at a strong edge
				if its.A[r, c] == 1:
					# we call the recursive function
					its.find_neighbors(r, c)

		# Remove remaining weak edges
		for r in range(its.rows-1):
			for c in range(its.cols-1):
				if its.A[r, c] == 0.5:
					its.A[r, c] = 0

		return its.A

	def find_neighbors(its, r, c):
		# Bounds checking
		if 0 <= r < its.rows and 0 <= c < its.cols:
			# if we haven't been here yet
			if not its.visited[r, c]: 
				# we have now
				its.visited[r, c] = True	

				# perhaps we're at a neighbor, is it connected? 
				if its.A[r, c] >= 0.5: 
					# weak neighbors become strong
					its.A[r, c] = 1

					# check its 8 neighbors
					its.find_neighbors(r-1, c-1) # top left
					its.find_neighbors(r-1, c+0) # top middle 
					its.find_neighbors(r-1, c+1) # top right 
					its.find_neighbors(r+0, c-1) # center left 
					its.find_neighbors(r+0, c+1) # center right
					its.find_neighbors(r+1, c-1) # bottom left 
					its.find_neighbors(r+1, c+0) # bottom middle
					its.find_neighbors(r+1, c+1) # bottom right

	def display(its, A):
		pylab.imshow(A, cmap = "gray")
		pylab.show()

if __name__ == '__main__':
	image = '../_ref/' + sys.argv[1]
	high = float(sys.argv[2])
	low = float(sys.argv[3])

	c = CannyEdgeDetector() 
	A = c.load(image)
	smooth_img = c.gaussian(A)
	edge_strength, edge_orientation = c.find_gradients(smooth_img)
	thinned_edge = c.nonmax_suppression(edge_strength, edge_orientation)
	image = c.hysteresis_thresholding(thinned_edge, high, low)
	
	c.display(image)