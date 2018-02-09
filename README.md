# Traffic-Sign-Classifier
Classifying German Traffic Sign Data

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set:

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset:

Here is an exploratory visualization of the data set. It is a bar chart for training data (Class vs Number of Images) showing how the data is disributed across various classes

[image1]: bar_chart.png "Training Image"
![alt text][image1]

### Design and Model Architecture

#### 1. Pre-process Techniques:
As a first step, I decided to add transformed images for classes with lower than 500 images.
I used getAffineTransform() from cv2 to transform the images. I transformed each image in lower class images four times.

Then, I converted the images to grayscale because I thought that the model would not gain a lot knowledge using colors. Also, it would save unnecessary computation power.

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

[image2]: trained0.png "Original Image from training data"
![alt text][image2]

[image3]: transformed1.png "Transform 1"
![alt text][image3]

[image4]: transformed2.png "Transform 2"
![alt text][image4]

[image5]: transformed3.png "Transform 3"
![alt text][image5]

[image6]: transformed4.png "Transform 4"
![alt text][image6]


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
|	Gray					|	32x32x1 Gray Image						|
|	Normalized		| 32x32x1 Normalized Image												|
| Convolution  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
|	Dropout				|	0.95 keep rate								|
| Max pooling	  | 2x2 stride,  outputs 14x14x6				|
| Convolution  	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
|	Dropout				|	0.85 keep rate								|
| Max pooling	  | 2x2 stride,  outputs 5x5x6				|
| Flatten     	|	outputs 400							|
|	Fully Connected| outputs 120     			|
|	RELU			    |												|
|	Fully Connected| outputs 84					|
|	RELU			    |												|
|	Fully Connected| outputs 43   			|
| Softmax				|        									|
