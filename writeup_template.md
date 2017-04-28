# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 - recording 1st track of my vehicle driving autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is similar to model that Nvidia used in End to End Learning for Self-Driving Cars (https://arxiv.org/pdf/1604.07316v1.pdf) except for one dropout layer inserted before flatten layer and one cropping layer inserted after normalization layer :

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Normalization | outputs 160x320x3 |
| Cropping | top cropping 70, bottom cropping 25, outputs 65x320x3 |
| Convolution 5x5 (kernel) | 2x2 stride (subsample), valid padding, outputs 31x158x24 |
| RELU					|												|
| Convolution 5x5	| 2x2 stride, valid padding, outputs 14x77x36 				|
| RELU					|
| Convolution 5x5	| 2x2 stride, valid padding, outputs 5x37x48 				|
| RELU					|
| Convolution 3x3	| 1x1 stride, valid padding, outputs 3x35x64 				|
| RELU					|
| Convolution 3x3	| 1x1 stride, valid padding, outputs 1x33x64 				|
| RELU					|
| Dropout					|	50% dropout											|
| Flatten	      	| outputs 2112 				|
| Fully connected		| outputs 100        									|
| Fully connected		| outputs 50        									|
| Fully connected		| outputs 10        									|
| Fully connected		| outputs 1        									|

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layers in order to reduce overfitting (model.py lines 111). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 119). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I used a combination of clockwise center lane driving, counter clockwise center lane driving, smoothly around curve driving (with flipped image augmentation), recovery driving from side (with flipped image augmentation). 

Images from left, center and right camera are used. Steer angle compensation with +0.2 and -0.2 is used for image from left and right camera, respectively.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use LeNet neural network model (http://yann.lecun.com/exdb/lenet/) with normalization inserted in first layer. However, it doesn't work well and I noticed that the output depth of convnet layer is shallow (depth = 6) for a 65x320x3 input and 5x5 kernel. This problem is not simple task as to recognize handwritten characters that may require few features. So I inferred that models with more deeper convnet may be more appropriate transfer learning candidate for this problem.

Then I tried Nvidia's model in End to End Learning for Self-Driving Cars (https://arxiv.org/pdf/1604.07316v1.pdf). I chose this model because it's a verified model for autonomous vehicle lane keeping. The output depth of convnet layer goes to 64, and it seems deep enough to capture some features more complex.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-116) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded couter-clockwise lap with center lane driving to compansate tendency of mostly left turn in clockwise lap.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side if it steered away from lane center.

These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]




To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]





After the collection process, I had 35126 number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by saturated validation loss in 5 epochs training.



I used an adam optimizer so that manually training the learning rate wasn't necessary.
