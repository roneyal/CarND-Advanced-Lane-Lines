## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib.png "Undistorted"
[image2]: ./output_images/calib_road.jpg "Undistorted road image"
[image3]: ./output_images/thresholds.jpg "Thresholds"
[image4]: ./output_images/transformed.jpg "Warp Example"
[image5]: ./output_images/poly.jpg "polynomial fit"
[image6]: ./output_images/lane.jpg "Output"
[video1]: ./out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the camera_calib.py. 

When the camera calib class is loaded, it performs calibration based on the chessboard images provided, and the camera matrix is saved (calibrate method in camera_calib.py, line 16).
The calibration is done by loading each image and finding the chessboard corners in that image. All the corners which are successfully found in the images are saved in the image points set (line 42), which is then calibrated against the object points (a set of equally spaced 2d points) (line 46 in camera_calib.py).

The camera matrix and distortion coefficients are saved and used for undistorting images (undistort method in line 50). Here is an example on one of the chessboard images:

![alt text][image1]


### Pipeline (single images)

The pipeline for a single image is implemented in high level in lane_finder.py in handle_image method (line 21).


#### 1. Provide an example of a distortion-corrected image.

The first step in the pipeline is applying the undistort method on the test image. Here is an example of before and after distortion correction on a test image:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The next step is applying color transforms, gradients and thresholds to convert the image to a binary thresholded image.
This is done in gradient_thresholds.py. To tune the different thresholds I plotted them all together, along with the original image and a combined filter (both seen on the left side of the plot).
After some exploration I got the best results by combining an X axis gradient filter, Y axis gradient filter, gradient magnitude filter, R channel filter and S channel filter (lines 73 - 100 in gradient_thresholds.py).
An example plot can be seen below:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The image perspective transform is done in perspective_transform.py. 
I identified the following points as the source:
[220, 720], [590, 450], [690, 450], [1060, 720]
And the following for the destination:
[300, 720], [300, 0], [950, 0], [950, 720]

An example transformation can be seen in the next image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window approach to find the lane lines in the transformed image. The initial lane is found by the method find_initial_lanes in sliding_window.py (line 12).
When there is a high confidence detection, the search for the next lane is done only in a window around the previous finding. This is done in the method find_next_line in sliding_window.py (line 83).
The lit pixels in each window are then used to fit a polynomial for each lane (lines 72 - 79 and 99 - 105 in sliding_window.py).
Here is a nice example of the fitted polynomials on the warped image:

![alt text][image5]

To determine if a fit on an image is a high confidence match or not, I used a basic heuristic based on a-priori knowledge of the lane shape. A high confidence detection was defined as one where  the lane lines are not too close (less than 315 cm on average) and not too far away (more than 400 cm on average), with a small standard deviation in the lane distances (less than 50cm) and where the car is identified to be close to the center (up to 50cm). In case the finding is not a "high confidence" detection, the next image is processed like the first image - the search for the lane lines is done on the whole width. See line 52 in lane_finder.py.

In addition, to smooth the noise between images, I used an alpha filter for the detections - high confidence detections were averaged with the previous findings with an alpha value of 0.1 and low confidence detections did not change the running average at all (alpha value of 0) - see lines 56-57 in lane_finder.py.
	  
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature radius is calculated in the method calc_radius in lane_finder.py.
I found it best to average both polynomials (right and left lane) and then to find the radius (instead of averaging two radius values). See line 105 - 127 in lane_finder.py.
The vehicle distance from center is calculated by the location of the lanes at the bottom of the image and the distance of each lane from the center pixel.
The distance in pixels is multiplied by 3.7/7 to get the distance in cm, and this is the value that is displayed on the output image. See lines 62-63 in lane_finder.py.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is a nice example of the lane drawn on the road. Since the car is slightly towards the left side of the lane, the distance from center is negative.
This was plotted using lines 68-79 in lane_finder.py.
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/y3k2-xK02J4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It was very difficult to tune the values for the filters in a robust manner. I believe it won't generalize well for other lighting and road conditions. Perhaps a voting algorithm between many filters would perform better and be more robust than a simple "or" between all filters.

In addition, my heuristics for determining detection confidence will fail in case the lane dimensions change (very wide or narrow lanes). 
To be more robust, the heuristics could be expanded, but this could hurt performance. Another option is to dynamically fit the expected lane widths according to a map or to a running average of previous observations.

