
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/sample_predictions.png
[image9]: ./examples/heatmap_filtered_image.png
[image10]: ./examples/example_1.png
[image11]: ./examples/example_2.png
[image12]: ./examples/example_3.png
[image13]: ./examples/example_4.png
[image14]: ./examples/example_5.png
[image15]: ./examples/example_6.png
[video1]: ./project_video_out.mp4.zip

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell "In[6]" of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In code cell "In[6]", I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

In code cell "In[6]", I tried various combinations of parameters and decided:
p_color_space='YCrCb'
p_hog_orient=9
p_hog_pix_per_cell=8
p_hog_cell_per_block=4
p_feature_types=['hog']

"p_hog_cell_per_block=4" provides me a smooth heatmap over the car windows. Smaller p_hog_cell_per_block=2 yields discountinous heatmap windows.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell "In[9]", I trained a linear SVM using only HOG features over YCrCb color space. The color features are not useful since the training accuracy is already 100%, dev and test accuracy close to 100%. Results shown in code cell "In[10]".

I also tried BaggingClassifier and RandomForest, yet the accuracy returned was lower.

Prior to training, I scale the feature to 0 mean and unit variant.

![Sample predictions][image8]


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In code cell "In[11]" and "In[12]", I execute window search over various size windows and search locations:

[(window_size_y, window_size_x) , (search_span_min_y, search_span_max_y),  (search_span_min_x, search_span_max_x)] = [

    ((128, 128), (380, 560), (790, 1280)),
    ((128, 256), (380, 560), (790, 1280)),
    ((64, 128),  (380, 470), (790, 1280)),
    ((96, 192),  (380, 490), (790, 1280)),
    ((96, 128),  (380, 490), (790, 1280)),
    ((264, 264),  (350, 650), (850, 1280)),
    ((200, 330),  (370, 600), (850, 1280)),
    ((260, 300),  (370, 600), (950, 1280))
    
]

First, I rescale the image such that (window_size_y, window_size_x) window map to (64, 64), which is the size of my training images. I compute the hog features over the serach locations (search_span_min_y, search_span_max_y) (search_span_min_x, search_span_max_x). I step window 1 cell each step, i.e. 2 adjacent windows will overlap 3/4, and collect my hog features for the widow from the hog features previously generated for the whole search image.

I decided the parameters by looking up video frame by frame, and mapped the car size and car locations to my window size and search locations.

![alt text][image3]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I improve the clasify acurracy by augmented the fliping left-right training data. I also remove the color histogram from training features to avoid overfiting, since hog features already yield near 100% accuracy in train, dev and test sets.

I have also experimented L1, L2 regulizer and limit training iterations in Linear SVM to prevent overfitting.

Ultimately I searched on two scales using YCrCb 3-channel HOG features.  Here are some example images:

![alt text][image4]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4.zip)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap. I applied Gausian blur as a voting mechanism accross windows to combine overllapping bounding boxes. I thresholded that Gausian blur heatmap to identify vehicle positions and filter false positives. 

I code cell "In[90]", I have also implemented exponential decay to smooth the heatmap accross frames.

Here's an example result showing the heatmap, the filtered image, and the image with bouding box from a series of frames of video:

![Top left: Heat map. Top right: Thresholded on heatmap. Bottom: Original image with bounding box][image9]


### Here are six frames and their corresponding heatmaps and binary thresholded:


![Example 1][image10]


![Example 2][image11]


![Example 3][image12]


![Example 4][image13]


![Example 5][image14]


![Example 6][image15]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further:

1. I have the training data of car and non-car images; I then train a Linear SVM to classify car image. The positive training mostly contain the full image of the back, the right and the left of the car. If a car is comming to the camera frame, and half of it appear in the image, the classifier would face dificulties in detecting half car window, since it never observes it in the training data. I have attempted to augmented half car image, yet has not been able to training it due to computation limitation. With more computation power, I can train the classifier on augmented data.

2. I perform window search with stepping 1 cell at a time. With stepping, the search space becomes non-continous, and the results depends on if I can step corectly on a window containing a car. This approach yield trade off between the size of the step and the precision of the prediciton. The classifier will perform well if I luckly step on a window having the full car image similar to the training examples.

On the other hand, I also search various car sizes pre-selected. Yet, in practice, we may not be able to pre-select all car sizes.

Therefore, I would suggest trying a better models independent from window size and step size such as Regional based Convoultional Network.

3. Sequence of frames has not been utilized in my approach; I could employ them into a deep learning model architecture such as Recurent Network to capture a vector representing a frame.

4. I have used the depth signals, and other location sensor data in this work. Sensor data with depth construction from LiDar would enable higher precisions.

