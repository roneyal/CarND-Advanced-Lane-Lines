import camera_calib
import gradient_thresholds
import perspective_transform
import sliding_window
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy #for saving images

class video_flow():

    def __init__(self):
        self.cam_calib = camera_calib.get_camera_calib()
        self.count = 0
        self.left_fit = None
        self.right_fit = None
        self.high_conf = False


    def handle_image(self, img):

        #scipy.misc.imsave('input_images/' + str(self.count) + '.jpg', img)
        undist = self.cam_calib.undistort(img, plot=False)
        thresh = gradient_thresholds.apply_all_thresholds(undist, plot=False)
        warper = perspective_transform.perspective_transform()
        warped = warper.warp(thresh)

        window = sliding_window.sliding_window()
        if self.high_conf == False:
            left_fit,right_fit = window.find_initial_lanes(warped)
            if self.count == 0:
                self.left_fit, self.right_fit = left_fit, right_fit
        else:
            left_fit, right_fit = window.find_next_line(warped, self.left_fit, self.right_fit)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        mean = np.mean(right_fitx - left_fitx) * 3.7 / 7
        std = np.std(right_fitx - left_fitx) * 3.7 / 7
        max = np.max(right_fitx - left_fitx) * 3.7 / 7

        lane_center_pixel = warped.shape[1] / 2
        vehicle_center_pixel = (left_fitx[-1] + right_fitx[-1]) / 2
        lane_dist = (lane_center_pixel - vehicle_center_pixel) * 3.7 / 7

        self.high_conf = True
        alpha = 0.2

        if mean < 315 or mean > 400 or std > 50 or np.abs(lane_dist) > 50:
            self.high_conf = False
            alpha = 0

        self.left_fit = (1.0 - alpha) * self.left_fit + alpha * left_fit
        self.right_fit = (1.0 - alpha) * self.right_fit + alpha * right_fit

        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        vehicle_center_pixel = (left_fitx[-1] + right_fitx[-1]) / 2
        lane_dist = (lane_center_pixel - vehicle_center_pixel) * 3.7 / 7

        left_curverad, right_curverad, center_curverad = calc_radius(ploty, left_fitx, right_fitx)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = warper.unwarp(color_warp)  # cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        #text = 'left radius: %dm, right radius: %dm' % (left_curverad, right_curverad)
        text = 'radius: %dm' % (center_curverad)
        text2 = 'distance from center: %dcm' % (lane_dist)
        #text3 = 'mean width %dcm, std. %dcm, max %dcm, high confidence: %s' %(mean, std, max, self.high_conf)


        result = cv2.putText(result, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        result = cv2.putText(result, text2, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        #scipy.misc.imsave('output_images/' + str(self.count) + '.jpg', result)
        self.count = self.count+1

        return result

    def plot_polynomial_fits(self, left_fitx, ploty, right_fitx):
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

def calc_radius(ploty, leftx, rightx):

    y_eval = np.max(ploty)

    centerx = (leftx + rightx) / 2

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    center_fit_cr = np.polyfit(ploty * ym_per_pix, centerx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    center_curverad = ((1 + (2 * center_fit_cr[0] * y_eval * ym_per_pix + center_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * center_fit_cr[0])

    return left_curverad, right_curverad, center_curverad


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def run_video():

    vid = video_flow()
    output = 'out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    out_clip = clip1.fl_image(vid.handle_image)  # NOTE: this function expects color images!!
    out_clip.write_videofile('out.mp4', audio=False)



if __name__ == '__main__':

    run_video()