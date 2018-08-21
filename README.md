# Vehicle Detection

This project build a software pipeline to detect vehicles in a video using different method including SVM+Sliding Window and YOLO.Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Estimate a bounding box for vehicles detected.
* Implement YOLO

[//]: # "Image References"
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/window_scale1.png
[image4]: ./output_images/window_scale1.5.png
[image5]: ./output_images/window_scale2.png
[image6]: ./output_images/window_scale2.5.png
[image7]: ./output_images/all_windows.png
[image8]: ./output_images/all_detections_single_frame1.png
[image9]: ./output_images/all_detections_single_frame2.png
[image10]: ./output_images/all_detections_single_frame7.png
[image11]: ./output_images/heatmap_frame1.png
[image12]: ./output_images/labeld.png
[image13]: ./output_images/detected.png
[Image14]: ./output_images/yolo.png

### Histogram of Oriented Gradients (HOG)

The code for this step is in lines 198 through 217 in the function `def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)` from the file called `train.py`, which I used to train the linear SVM classifier. I extract the HOG feature by using `skimage.feature.hog` and return the feature vector(the HOG features on the image are flattened).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I tried various combinations of parameters and finally kept using `oritent=9`, `pixels_per_cell=9`, `cells_per_block=2`.

I trained a linear SVM using feature combination from binned color features, histogram of colors and HOG feature. I tried all of the color space in the range of 'RGB, HSV, LUV, HLS, YUV, YCrCb'. However, based on the SVM accuracy and the vehicle detection on the video images with smallest noise, I used 'LUV' as the color space.  After extracting the features from both car and noncar images, I combine and shuffle then(code line from 59 through 63 in `train.py`). Before fitting to the SVM classifier, I normalized the data from code line 66 through 68 in file `trian.py`.

### Sliding Window Search

I decided to search window with sizes from four scales 1, 1.5, 2, 2.5(these scales are used in line 53 and line55 in file `Vehicle_Detector.py`, actually used for scale the image while I always use the window size 64 on different scaled images), all over the lower half of the image and came up with these windows:
##### Scale 1
![alt text][image3]
##### Scale 1.5
![alt text][image4]
##### Scale 2
![alt text][image5]
##### Scale 2.5
![alt text][image6]
##### All the searching windows
![alt text][image7]
I also fixed the step size to be 2 cells(16 pixels) and since the window size is also fixed at 64(note I just scale the image with fixed window size to get the effect that I have different window size), the overlap is always 1/4 of the window.

#### 

Ultimately I searched on four scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]

This is the detection result from one single frame. We can see most of the bounding boxes are around the vehicles except for two outliers. The code for this step is from line 46 to line 146 in the function `def get_cur_rects(self, img_org)` in file `Vehicle_Detector.py`.

Some other results:
![alt text][image9]
![alt text][image10]

To reduce some false positives, I actually just limit the searching range of different scales. For instance, I found the small window might bring a lot of false positives int the lower part of the image, I just set the searching range of the smaller window to be higher in the image.

---

### Video Implementation

Here's a [link to my video result](https://youtu.be/MQZySgs-ncw)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 

Here's an example result showing the heatmap from a series of frames of video:
![alt text][image11]
This is the heat map corresponding to this frame. The code for this step is from line 148 to line 159 in function `def update_heatmap(self, cur_rect)` in file `Vehicle_Detector.py`. 

In order to remove the outliers, I implement a queue to store the bounding boxes in several consecutive frames(variables are `self.rects_queue_length = 10` and `self.rects_queue = []`). I maintain the size of the queue with size `self.rects_queue_length` and at the very beginning I just keet pushing rects until the size of the queue reach the length. After that for each frame, I just pop out the rects in the front, and push the new rects to the back. Also I update the heat map by reducing 1 from all of the rects I removed and increasing 1 from all of the new rects. The code for this part is in function `def update_heatmap(self, cur_rect:` and `def get_vehicle_bounding_boxes(self)` from line 148 through line 181 in file `Vehicle_Detector.py`.

Finally I get the vehicle bounding boxes by taking points from the heat map with the value higher than the threshold(I tuned to `self.heatmap_thresh = 30` when `self.rects_queue_length = 10`).  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Code for this part is from line 161 through 181 in function `def get_vehicle_bounding_boxes(self)` in file `Vehicle_Detector.py`.

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap after 10 consecutive frames:
![alt text][image12]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]

---

### YOLO Implementation

1. Download Pascal VOC dataset, and create correct directories

```Shell
$ ./download_data.sh
```
2. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing) weight file and put it in `data/weight`

3. Modify configuration in `yolo/config.py`
  
4. Test
  ```Shell
  $ python test_yolo.py
  ```

### Requirements
1. Tensorflow

2. OpenCV

![alt text][image14]
