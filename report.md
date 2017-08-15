## Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


---
###Writeup / README

For the "final" notebook please refer to `model-cheat.ipynb`. This notebook has probably been "overtrained" (i.e. using information that the car is always on the leftmost lane) and may not generalise well. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

A large portion of code used in this project was fully contained within the lecture videos. The functions could simply be copied over from there and used. 

For the HOG features, this was created in cell 2. 

I started by reading in all the `vehicle` and `non-vehicle` images, using the `get_hog_features`.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes with parameters `orient=9`, `pix_per_cell=8`, `cell_per_block=8`:

![alt text](report/car_hog.png)

![alt text](report/ncar_hog.png)

Next I explored a slightly different parameter set which is the one which I ended up using, `orient=11`, `pix_per_cell=10`, `cell_per_block=2`

![alt text](report/car_hog2.png)

![alt text](report/ncar_hog2.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and through the scikit-learn pipeline and settled on the set which provided reasonably good score on the training set. I found that the first set created too many features and failed to generalise as well as the second set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using scikit-learn pipeline. I created a custom `Transformer` using the function `single_img_features`. This allowed me to easily change some of the parameters within this function to assess the impact on the machine learning pipeline. 

As this could be viewed as a hyper parameter set, it was trained through a variation of a grid search (i.e. iterate through a set of parameters which feed into `FeatureCreator` and assess the training rate). Different learners were assessed, but there was minimal improvement when switching among them (i.e. linear svm was "good enough")

![pipeline](report/pipeline.png)

For example `YUV` and `YCrCb` seemed to get performance ~0.79-0.88 accuracy range where as using `RGB` yielded ~0.96-0.99 accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code was implemented as per code in the lessons.

The window positions were assessed at various windows sizes being:

*  64x64
*  96x96
*  128x128
*  192x192
*  256x256



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

We generally chose the `y_start_stop` parameter to be around half way down the image (`350`), so to not consider objects in the "sky" (i.e. above the horizon)

For the smaller windows, we considered only between 350-500px height whilst on the larger window sizes we considered 350-720px height.

In the first cut it looked like this

![alt text](report/window1.png)

To reduce the false positive rate, and noting that all the images are to the rigth of the car, we can restrict the search space (this is kind of cheating, but in the sample videos the car was always on the left most lane)

![alt text](report/window2.png)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_20170815_v2.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The filter used was the same as shown in the videos. The `apply_threshold` was applied to identify vehicle positions

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



### Here are six frames and their corresponding heatmaps:

![alt text](report/heatmap.png)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text](report/heatmap_label.png)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text](report/final_process.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The final video was done via "cheating" as I know in the video the car never moves from the left most lane we simply ignored everything on the left. 

Further improvements could be using state of the art approaches to do object recognition such as YOLO algorithm. 

It has also seemed to not handle cars coming from the other direction very well and also the barriers - we will probably have to do some more training there. My hypothesis is since the barrier in the middle is a single solid colour, that might be the pattern that is being picked up by the SVM.
