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

[image1]: images/bar_chart.png "Training Image"
![alt text][image1]

### Design and Model Architecture

#### 1. Pre-process Techniques:
As a first step, I decided to add transformed images for classes with lower than 500 images.
I used getAffineTransform() from cv2 to transform the images. I transformed each image in lower class images four times.

Then, I converted the images to grayscale because I thought that the model would not gain a lot knowledge using colors. Also, it would save unnecessary computation power.

Then, I normalized the images.

Here is an example of an original image and augmented images:

[image2]: images/trained0.png "Original Image from training data"
![alt text][image2]

[image3]: images/transformed1.png "Transform 1"
![alt text][image3]

[image4]: images/transformed2.png "Transform 2"
![alt text][image4]

[image5]: images/transformed3.png "Transform 3"
![alt text][image5]

[image6]: images/transformed4.png "Transform 4"
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

#### 3. Hyperparameters:

To train the model, I used an EPOCHS of 25, BATCH SIZE of 128, learning rate of 0.002

#### 4. Approach taken for finding solution and Accuracy Score:

I used LeNet architecture as it has been widely used for recognizing handwritten digits and gives good accuracy with decent amount of computation power. I think it should give good accuracy for traffic sign recognition with some modifications.

I added two dropout layers, each after Convolution layer in LeNet5.

I started with standard learning rate of 0.001, epochs of 10 and batch size of 256. First, I changed the batch size, keeping other values same as before, I got better accuracy using batch size of 128. Subsequently, after various combinations the learning rate of 0.002 and epochs of 25 gave better accuracy than before.

The accuracy on my final model were:
* validation set accuracy of 0.939
* test set accuracy of 0.917

### Test a Model on New Images

#### 1. German traffic signs I found on the web:

Here are six German traffic signs that I found on the web:

[img1]: images/img1.jpg "MyTestImage 1"
![alt text][img1]

[img2]: images/img2.jpg "MyTestImage 2"
![alt text][img2]

[img3]: images/img3.jpg "MyTestImage 3"
![alt text][img3]

[img4]: images/img4.jpg "MyTestImage 4"
![alt text][img4]

[img5]: images/img5.jpg "MyTestImage 5"
![alt text][img5]

[img6]: images/img6.jpg "MyTestImage 6"
![alt text][img6]

#### 2. Model's predictions on new traffic signs:

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Entry     			| No Entry 							|
| Speed limit 50km/h| Speed limit 50km/h				|
| Yield         		| Yield    			 				|
| Turn left ahead   | Keep right    							|
| Speed limit 70km/h| Speed limit 70km/h  	|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%.
