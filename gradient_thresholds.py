import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def apply_threshold(img, thresh=(0,255)):
    masked = np.zeros_like(img)
    masked[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return masked

def apply_sobel(gray, orient='x', sobel_kernel=3):
    if orient == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return grad

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = apply_sobel(gray, orient, sobel_kernel)
    #plt.imshow(grad)
    grad_binary = apply_threshold(grad, thresh=thresh)
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = apply_sobel(gray, 'x', sobel_kernel)
    grad_y = apply_sobel(gray, 'y', sobel_kernel)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Apply threshold
    mag_binary = apply_threshold(magnitude, mag_thresh)
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = apply_sobel(gray, 'x', sobel_kernel)
    grad_y = apply_sobel(gray, 'y', sobel_kernel)
    angles = np.arctan2(grad_y, grad_x)
    # Apply threshold
    dir_binary = apply_threshold(angles, thresh)
    return dir_binary

def hls_threshold(image, channel='S', thresh=(0,255)):
    #Change color map
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if channel == 'H':
        channel_idx = 0
    elif channel == 'L':
        channel_idx = 1
    else:
        channel_idx = 2
    single_channel = hls[:,:,channel_idx]
    binary = np.zeros_like(single_channel)
    binary[(single_channel > thresh[0]) & (single_channel <= thresh[1])] = 1
    return binary

def rgb_threshold(image, channel='R', thresh=(0,255)):
    #Change color map
    if channel == 'R':
        channel_idx = 0
    elif channel == 'G':
        channel_idx = 1
    else:
        channel_idx = 2
    single_channel = image[:,:,channel_idx]
    binary = np.zeros_like(single_channel)
    binary[(single_channel > thresh[0]) & (single_channel <= thresh[1])] = 1
    return binary

def apply_all_thresholds(image, plot = False):
    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(250, 3000))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(500, 3000))
    # grady = np.zeros_like(grady)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(700, 1000))
    dir_binary = np.zeros_like(mag_binary)
    #dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(np.pi * 0.25, np.pi * 0.3))
    #dir_binary[dir_threshold(image, sobel_kernel=3, thresh=(np.pi * 0.75, np.pi * 0.8)) == 1] = 1
    #dir_binary[dir_threshold(image, sobel_kernel=3, thresh=(-np.pi * 0.75, -np.pi * 0.65)) == 1] = 1
    #dir_binary[dir_threshold(image, sobel_kernel=3, thresh=(-np.pi * 0.30, -np.pi * 0.15)) == 1] = 1
    s_binary = hls_threshold(image, 'S', thresh=(90, 255))
    r_binary = rgb_threshold(image, 'R', thresh=(200,250))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1)) |
             ((mag_binary == 1)
              #& (dir_binary == 1)
              ) |
             (r_binary == 1) |
             (s_binary == 1)] = 1
    combined[670:, :] = 0 #mask the hood of the vehicle
    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((gradx == 1) | (grady == 1))] = 1
    combined_mag = np.zeros_like(dir_binary)
    combined_mag[((mag_binary == 1) & (dir_binary == 1))] = 1

    if plot == True:
        f, subplots = plt.subplots(3, 3, figsize=(24, 9))
        f.tight_layout()
        subplots[0, 0].imshow(image)
        subplots[0, 0].set_title('Original Image', fontsize=10)
        subplots[1, 0].imshow(combined, cmap='gray')
        subplots[1, 0].set_title('All Combined Thresholds', fontsize=10)
        subplots[2, 1].imshow(s_binary, cmap='gray')
        subplots[2, 1].set_title('Threshold S channel', fontsize=10)
        subplots[0, 2].imshow(gradx, cmap='gray')
        subplots[0, 2].set_title('Thresholded X Gradient', fontsize=10)
        subplots[1, 2].imshow(grady, cmap='gray')
        subplots[1, 2].set_title('Thresholded Y Gradient', fontsize=10)
        subplots[2, 2].imshow(combined_grad, cmap='gray')
        subplots[2, 2].set_title('Combined X and Y Gradient', fontsize=10)
        subplots[0, 1].imshow(mag_binary, cmap='gray')
        subplots[0, 1].set_title('Thresholded Gradient Magnitude', fontsize=10)
        subplots[1, 1].imshow(r_binary, cmap='gray')
        subplots[1, 1].set_title('Thresholded R channel', fontsize=10)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('output_images/thresholds.jpg')
        plt.show()

    return combined

if __name__ == '__main__':

    image = mpimg.imread('input_images/1.jpg')

    thresholded = apply_all_thresholds(image, True)