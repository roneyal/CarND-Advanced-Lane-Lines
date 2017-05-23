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





if __name__ == '__main__':


    # prepare object points
    nx = 9
    ny = 6

    images = glob.glob('camera_cal/calibration*.jpg')


    camera_calib = camera_calib(nx, ny, images)
    camera_calib.calibrate()

    img = cv2.imread('camera_cal/calibration2.jpg')
    camera_calib.undistort(img, plot=True)

    '''

    #object points for a single image
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    #object points for all images
    object_points = []
    #image points for all images
    img_points = [] #np.array(dtype=np.float32)

    for fname in images:

        img = cv2.imread(fname)
        #plt.imshow(img)
        print(img.shape)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            object_points.append(objp)
            img_points.append(corners)
            #print (object_points)


    img = cv2.imread('camera_cal/calibration2.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])

    object_points = np.array(object_points, dtype=np.float32)
    print(object_points.shape)
    #img_points = np.array(img_points, dtype=np.float32)
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img.shape[0:2], None, None)
    undist = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()
'''