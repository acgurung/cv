import cv2
import numpy as np, skimage, skimage.io, pylab, scipy.ndimage.filters, math
from skimage import data
from math import pi
from copy import deepcopy

class StereoMatch:
    def __init__(its):
        its.rows = 0
        its.cols = 0

    def load(its, image):
        A = skimage.io.imread(image)
        A = skimage.img_as_float(A)
        A = np.float32(A)

        its.rows = A.shape[0]
        its.cols = A.shape[1]
        its.shape = A.shape
        its.max_d = int(its.rows * (1/3)) 
        return A

    def find_dsi(its, A, B):
        dsi = np.zeros((its.rows, its.cols, its.max_d))

        for d in range(its.max_d):
            for r in range(its.rows):
                for c in range(its.cols):
                    t = c-d
                    if t < 0:
                        t = 0

                    dsi[r, c, d] = ((A[r, c, 0] - B[r, t, 0])**2 +
                                    (A[r, c, 1] - B[r, t, 1])**2 +
                                    (A[r, c, 2] - B[r, t, 2])**2)

        return dsi

    def spatial_aggregation(its, A, sigma):
        result = deepcopy(A)
        for d in range(its.max_d):
            result[:, :, d] = skimage.filters.gaussian(A[:, :, d], sigma=sigma)

        return result 

    def disparity_map(its, A):
        disparity_map = np.zeros((its.rows, its.cols))
        for r in range(its.rows):
            for c in range(its.cols):
                s = np.argmin(A[r, c])
                disparity_map[r, c] = s

        return disparity_map
    
    def bilateral_filter(its, A, dsi):
        dsi = np.float32(dsi)

        joint = A
        source = dsi
        diameter = 2
        sigma_color = 400
        sigma_space = 400

        result = np.zeros((its.rows, its.cols, its.max_d))

        for d in range(its.max_d):
            result[:,:,d] = cv2.ximgproc.jointBilateralFilter(joint, source[:, :, d], diameter, sigma_color, sigma_space)

        bilateral_filter = its.disparity_map(result)
        return bilateral_filter

    def left_right_check(its, A, B, ground_truth, d_map):
        # left-right consistency check
        threshold = 10
        occlusion = np.zeros((its.rows, its.cols))
        dsi = np.zeros((its.rows, its.cols, its.max_d))
        
        for d in range(its.max_d):
            for r in range(its.rows):
                for c in range(its.cols):
                    m = c+d
                    if m >= its.cols:
                        m = c

                    dsi[r, c, d] = ((A[r, m, 0] - B[r, c, 0])**2 +
                                    (A[r, m, 1] - B[r, c, 1])**2 +
                                    (A[r, m, 2] - B[r, c, 2])**2)

        sa = its.spatial_aggregation(dsi, sigma=0.2)

        d_map2 = its.disparity_map(sa)

        for r in range(its.rows):
            for c in range(its.cols):
                result = d_map2[r,c]-d_map[r,c]

                if result > threshold: # mismatch
                    occlusion[r,c] = 0
                else:
                    occlusion[r,c] = d_map2[r,c]

        sum_o = 0
        for r in range(its.rows):
            for c in range(its.cols):
                if occlusion[r,c] == 0:
                    pass
                else:
                    sum_o += (ground_truth[r, c] - occlusion[r, c])**2

        sum_o /= (its.rows * its.cols)
        # print(math.sqrt(sum_o))
        return occlusion

    def root_mean_square(its, ground_truth, d_map):
        print(np.sqrt(np.mean((ground_truth-d_map)**2)))

    def display(its, image):
        pylab.imshow(image, cmap='jet')
        pylab.show()

def ssd(A,B):
    squares = (A[:,:,:3] - B[:,:,:3]) ** 2
    return np.sum(squares)

if __name__ == '__main__':
    # Load the left and right camera
    left_img = 'im1_left.png'
    right_img = 'im1_right.png'

    s = StereoMatch()
    a = s.load(left_img)
    b = s.load(right_img)

    dsi = s.find_dsi(a, b)
    sa = s.spatial_aggregation(dsi, sigma=0.7)
    dm = s.disparity_map(sa)

    # Load the ground truth 
    gt = 'gt/gt.npy'
    ground_truth = np.load(gt)
    s.root_mean_square(ground_truth, dm)

    bf = s.bilateral_filter(a, dsi)
    s.root_mean_square(ground_truth, bf)

    o = s.left_right_check(a, b, ground_truth, dm)
    s.root_mean_square(ground_truth, o)

    s.display(o)