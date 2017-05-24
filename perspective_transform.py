import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class perspective_transform():

    def __init__(self):
        src = np.float32([[220, 720], [590, 450], [690, 450], [1060, 720]])
        dst = np.float32([[300, 720], [300, 0], [950, 0], [950, 720]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warped

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)
        return warped


if __name__ == '__main__':


    img = mpimg.imread('test_images/straight_lines2.jpg')

    perspective_transform = perspective_transform()

    warped = perspective_transform.warp(img)

    f, subplots = plt.subplots(1,2, figsize=(24, 9))
    f.tight_layout()

    subplots[0].imshow(warped)
    subplots[1].imshow(img)
    plt.show()