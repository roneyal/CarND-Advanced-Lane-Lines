import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class camera_calib():

    def __init__(self, nx, ny, image_files):
        self.nx = nx
        self.ny = ny
        self.image_files = image_files

    def calibrate(self):
        # object points for a single image
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # object points for all images
        object_points = []
        # image points for all images
        img_points = []  # np.array(dtype=np.float32)

        for fname in self.image_files:

            img = cv2.imread(fname)
            # plt.imshow(img)
            print(img.shape)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                object_points.append(objp)
                img_points.append(corners)
                # print (object_points)

        img = cv2.imread(self.image_files[0])
        retval, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img.shape[0:2],
                                                                             None, None)

    def undistort(self, img, plot = False):
        undist = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(undist)
            ax2.set_title('Undistorted Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

            plt.show()


    # Define a function that takes an image, number of x and y points,
    # camera matrix and distortion coefficients
    def corners_unwarp(img, nx, ny, mtx, dist):
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            # If we found corners, draw them! (just for fun)
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            offset = 100  # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])

            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                              [img_size[0] - offset, img_size[1] - offset],
                              [offset, img_size[1] - offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)

        # Return the resulting image and matrix
        return warped, M


if __name__ == '__main__':


    # prepare object points
    nx = 9
    ny = 6

    images = glob.glob('camera_cal/calibration*.jpg')


    camera_calib = camera_calib(nx, ny, images)
    camera_calib.calibrate()

    img = cv2.imread('camera_cal/calibration2.jpg')
    camera_calib.undistort(img, plot=True)

