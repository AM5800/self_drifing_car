# Problem overview
As part of the [Udacity self-driving car nanodegree program](http://udacity.com/drive) I was given a task to write an AI capable of driving car in a specially built simulator. Position and orientation of the car - those are parameters unknown to the model. It only allowed to see an image from the frontal camera mounted on the car. Given an image from a camera and constant throttle it should predict steering wheel angle. 

Here is what my result looks like
https://www.youtube.com/watch?v=eJDFs58AD04

# Solution approach
This problem is approached in terms of supervised learning. I.e we build some model(models), feed it sample images with marked steering angle and hope it will be able to drive. In this particular environment we have an ability to generate data by ourselfs - by driving a car in a simulator while recording required data. So the first step is to gather training data

# Dataset creation
I use two separate datasets in this project. Fist, "training set" is used to train several models. Second, "validation set" is used to evaluate model's performance and choose best model. And finally I can always launch simulator and let my model drive.

## Training set
To create a training set one can simply drive both provided test tracks in the center. But this is not enough: if a model leaves perfect trajectory (and it will) - it won't know what to do. That's why I also add some recovery maneuvers:
I move car to the edge of road and back. But then I delete frames where car is moving towards the edge. Since network sees our input and will eventually try to mimic our behaviour - it is a bad idea to show it how to get off the road. But if it happens - model will know what to do. Total training set size is 4992 images.

## Validation set
Early experiments showed that it is quite hard bo evaluate model based only on validation data. Two models with almost same validation score might behave totally different - one driving smoothly and ohter - driving in zigzags, loosing control and eventually leaving the test track. This makes automatic model selection very hard. 

To address this issue I first tried to increase size of the validation set. But it didn't help. I think this is because model needs only 2-3 frames to loose control. And even though mean square error for such frames will be high - it will be unnoticable when there are 6000 other frames which model had driven perfectly.

Solution that worked is to decrease validation set size. I was monitoring how models are driving on the test track and if I saw some place where it was frequently misbehaving - I made a few validation frames with expected steering angle. 
I ended with only 21 total images in validation set. Despite such small size my confidence in validation score is very high now. I am now sure that almost any model with validation score less than 0.1 is capable to drive. Which made automatic model selection very simple and predictable.

## Image preprocessing
I have tried to convert images to HSV colorspace. Intuition behind this desicion is that change in the lighting conditions will mostly affect one channel - V. And all three channels are usually affected in RGB model making it harder for model to learn the dependency between channels. 

Another technique is image normalization (R/255-0.5; G/255-0.5; B/255-0.5). In theory it should speed up model convergence because model has to spend less effort adapting for constantly changing distribution in the input data. In practice I haven't noticed any speed up in covergence.

# Training
I have tried lots of models with different parameters. But they have a lot in common:
- Models use Convolutional layers with relu activation
- Last layer contains single "node" - steering angle
- Mean square error is used as validation loss funciton

To minimize MSE I use Adam optimizer. And to find best network architecture and parameters I use grid search. In particular, with grid search I am optimizing:
- Network architecture
- Dropout value
- Use of Batch Normalization
- Image preprocessing technique

To reduce amount of RAM required to store the whole dataset, I store it as a list of paths to images. And when it comes to training I load only one batch of images with keras fit_generator.

Grid usually contains a lof of nodes. To speedup grid search even further I used early stopping. Also I am checking model after each epoch and if it has best global validation performance - I save it.

Next I will describe some notable networks

## AlexNet
I always start vision problems with a simple AlexNet-like network. In this case it starts with 4 convolution layers and ends with 2 hidden layers connected to regressor. Each convolution layer uses 'relu' activation and followed by max_pooling layer. There is also batch normalization layer between all layers. BN layer serves two purposes: first, it makes models to converge faster (and for really deep models like inceptionV3 it is super important). Second, it works as a regularizer. Since the mean and bias are learned from batches - net never sees the same input twice. Actual placement of BN layer is arguable, but I met recommendations to place them after each convolution layer.

Dropout also added to last hidden layer to even further prevent overfitting.

scheme_link

## AlexNet modifications
I also tried some modifications of alexnet. First, I have added dropout layers between convolutions. And second, I have removed max pooling layers. And increased convolution stride respectively.

## InceptionV3
Another interesting architecture that I tried - is inception network from google. You can see it's amazing architecture here. Keras framework has built-in function for creating this network keras.applications.inceptionv3
This network is used for image classification. So I have changed last layer to be regressor.

Mindblowing scheme is here

# Results
Default model predicts steering angle and drives with constant throttle. But this constant throttle is just not enough for test track 2 because it has some uphill parts. I have changed throttle according to this formula:

formula

In the end all described networks can drive a car in a simulator. I haven't found any difference betwenn alexnet modifications except that simple alexnet was training slightly faster on average.

Inception network showed good results on validation dataset. But very poor results in actual driving. It was always doing zig-zags which usually led it into the wall. One of the reasons for this might be the fact that it was taking 10 times more to process each frame. And because of this delay new steering angles come with too big delay. To proove that version I have added 0.09 second delay to a model that drives good. And it ended zigzaging too.

So final model uses alexnet architecture, batch normalization and dropout = 0.7

validation graph link

# Conclusion
Alexnet is able to drive both test tracks quite well. Inception network shows good "static" results but is unable to produce result fast enough to drive. 

This project was super useful for me. I have tried different neural network architectures and developed a lot of knowledge and intuition on how to train them.


# Appendix: project structure
* [model.py](model.py) - contains NN definition and functions to train it
* [drive.py](drive.py) - controls simulation
* [dataset.py](dataset.py) - utiliy functions for dataset loading and transformation
* [grid.py](grid.py) - utiliy functions for grid manipulations
* [model.json](model.json) - final model structure definition
* [model.h5](model.h5) - final model's weights
* [model.cfg](model.cfg) - stores dataset name which was used during training
* [model.png](model.png) - final model structure graph
