# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicles.jpg
[image2]: ./output_images/non_vehicles.jpg
[image3]: ./output_images/vehicle_hog.jpg
[image4]: ./output_images/non_vehicle_hog.jpg
[image5]: ./output_images/window_search.jpg
[image6]: ./output_images/heat.jpg
[image7]: ./output_images/car_position.jpg
[video1]: https://youtu.be/xh9sE1YCyyY

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed the first image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the following set of paramenters was efficient enough without giving up accuracy.

* `color_space = 'YCrCb'`
* `orient = 11`
* `pix_per_cell = 16`
* `cell_per_block = 2 `
* `hog_channel = "ALL"`

Originally I used `orient = 8` and `pix_per_cell = 8`, but it took too much time to compute HOG features with this setting, so I changed these parameters to speed up the process without compromising the accuracy.


####3 . Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, color features, and spatial features. The code for training and defining the classifier is contained in the third cell block in the IPython Notebook.

I specifically used `sklearn.model_selection.GridSearchCV` to try out different parameters and `train_test_split` to split the data into training and testing sets randomly.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window search method is contained in the fourth cell block. I tried different scales, and the combination of `[1.5, 1.7, 2.0]` gave the best result. With around 0.8 overlap windows.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.

Here's a [link to my video result](https://youtu.be/xh9sE1YCyyY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example of the heatmap with resulting bounding boxes

![alt text][image6]
![alt text][image7]

In order to produce stable bounding boxes in the video, for each frame, I used the heatmap from the previous 7 frames (added up) to determine where the bounding boxes should be.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As described above, I used a set a parameters that gave a very good result, but the algorithm was vary computationally-heavy. It took more than 10s to process a single frame!!!! This is not acceptable in a real-time environment.

Eventually, I found parameters that produced similar result but an efficient one. The current model can be processed 1 frame/s.

In the video, it is easy to find out that if the car is on the edge (right side). The algorithm may not be able to detect it. One possible improvement is to use more training data with half a car and body of cars. Also, some of the bounding boxes are not tight enough. This may cause problems in the real-world situation. 
