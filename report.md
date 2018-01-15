### **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_images.png
[image2]: ./output_images/sliding_window_search.png
[image3]: ./output_images/heat_map.png
[image4]: ./output_images/heat_map_1.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./result.mp4



#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/report.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

#### 2. My Approach
* My code can be found in [P5.ipynb](./P5.ipynb)
* My appraoch was to use a convolutional neural network to detect cars in the video
* I used the two data sets on the Udacity website taken from the GTI image database and the KITTI vision benchmark suite.
![alt text][image1]

* The data set has 17760 colour images all shaped 64x64
* I split the dataset into 90% for training and 10% for testing
* This is the CNN I used to train the network
```
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv1',input_shape=input_shape, padding="same"))  
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2',padding="same"))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv3',padding="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (8,8), activation="relu",name="dense1")) 
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (1,1), name="dense2", activation="tanh")) 

```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
* This is implemented in cell 9
* I decided to search within a region of the field of view of the camera on the car. This field of view removes the bonnet and the horizon. 
* Every image in that field of view is is passed through the trained classifier and a prediction is made whether its a car or not. If it is a bounding box is drawn around the car

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
by not using any fully connected layer at the end of the CNN I was able to apply a network trained on 64x64 images to bigger ones like the images coming from the camera on the car
Here are some example images:
![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions in cell 11. 
Here's an example result showing the heatmap from a series of frames of video and the bounding boxes then overlaid on the last frame of video:

![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. I would like to try other approaches to solving this like YOLO and SSD. I tried using YOLO initially, however I couldn't quite get it to work so I opted for a simple CNN
2. I would like to try another technique for the bounding boxes that makes use of previous information to gradually expand the box.


#### References
1. https://github.com/maxritter/SDC-Vehicle-Lane-Detection
