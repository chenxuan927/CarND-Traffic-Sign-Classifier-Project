# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class.jpg "Traffic Sign Classes"
[image2]: ./training.jpg "Training Dataset"
[image3]: ./validation.jpg "Validation Dataset"
[image4]: ./testing.jpg "Testing Dataset"
[image5]: ./test_image/go_straight_or_right.jpg "Traffic Sign 1"
[image6]: ./test_image/priority_road.jpg "Traffic Sign 2"
[image7]: ./test_image/road_work.jpg "Traffic Sign 3"
[image8]: ./test_image/speed_limit_30.jpg "Traffic Sign 4"
[image9]: ./test_image/stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32 , 32, 3)
* The number of unique classes/labels in the data set is 43

![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed across the various classes.

![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image data preprocessing, I only perform the minimally required normalization method following the given instruction. In details, for image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and is used in this project. 

The goal of normalization is to change the values of the dataset to a common scale, without distorting differences in the ranges of input values. It can prepare a "better" dataset for training.  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 					|
| Flattern	      		| outputs 400									|
| Fully connected		| output 120       								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 84       								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 43      								|
|						|												|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:
* Number of epochs = 30. 
* Batch size = 128
* Learning rate = 0.001
* Optimizer - Adam algorithm
* Dropout = 0.5 (for training set only)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of ? 
* test set accuracy of ?


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
  LeNet-5 architecture as provided in CarND-LeNet-Lab is used at first

* What were some problems with the initial architecture?

  the validation accuracy is always less than 0.9 and it gets overfitting as I increase the epochs number. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  drop out is applied at fully connected layer to prevent overfitting 

* Which parameters were tuned? How were they adjusted and why?

  Number of epochs is set to 30 to prevent overfitting, because increasing this parameter does not give any significant improvents 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  drop out is applied to prevent overfitting to help the model get a better result than before. 

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

All images are with watermark, which may bring some trouble to classification. All are converted to (32, 32) before being entering the network:

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or right  | Go straight or right  						| 
| Priority road    		| Priority road									|
| Road work				| Road work										|
| Speed limit (30km/h)	| Speed limit (30km/h)					 		|
| Stop					| Stop   										|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of .


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a go straight or right sign (probability of 0.64), and the image does contain a go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.63         			| Go straight or right  						| 
| 0.36     				| Keep right									|
| 0.01					| General caution								|
| 0.00	      			| Bumpy Road					 				|
| 0.00				    | Slippery Road      							|


For the second image, the model is very sure that this is a priority road sign (probability of 1.00), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road  								| 
| 0.00     				| Traffic signals								|
| 0.00					| Right-of-way at the next intersection			|
| 0.00	      			| Yield					 						|
| 0.00				    | Road work     								|

For the third image, the model is very sure that this is a road_work sign (probability of 1.00), and the image does contain a Priority Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| road_work  									| 
| 0.00     				| Dangerous curve to the right					|
| 0.00					| Bumpy road									|
| 0.00	      			| Slippery road				 					|
| 0.00				    | Bicycles crossing   							| 

For the fourth image, the model is very sure that this is a Speed limit (30km/h) (probability of 1.00), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)  						| 
| 0.00     				| Speed limit (70km/h)  						| 
| 0.00					| Speed limit (50km/h)  						| 
| 0.00	      			| Speed limit (20km/h)  						| 
| 0.00				    | Yield 										| 

For the fifth image, the model is very sure that this is a stop sign (probability of 1.00), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop 											| 
| 0.00     				| Speed limit (60km/h)							|
| 0.00					| Yield											|
| 0.00	      			| No vehicles			 						|
| 0.00				    | Bicycles crossing   							| 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
