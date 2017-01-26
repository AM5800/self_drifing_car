# Problem overview
As part of the Udacity Self-driving car nanodegree program I was given a task to write an AI capable of driving car in a specially build simulator. Position and orientation of the car - those are parameters unknown to the model. It only allowed to see an image from the frontal camera mounted on the car. Given an image from a camera and constant throttle it should predict steering wheel angle. 

Sample view from car camera

Video link

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

## Project structure
* [model.py](model.py) - contains NN definition and functions to train it
* [drive.py](drive.py) - controls simulation
* [dataset.py](dataset.py) - utiliy functions for dataset loading and transformation
* [model.json](model.json) - final model structure definition
* [model.h5](model.h5) - final model's weights
* [model.cfg](model.cfg) - stores dataset name which was used during training
* [model.png](model.png) - final model structure graph




## Dataset preparation
Dataset consists of training data and validation data. 
Test set is not generated because testing in the simulator effectively replaces it.
Training and validation dataset are stored separately. This is done to improve reproducibility of results

Only center images were used and I didn't use second track in training. This was done intentionally to check how well model can handle absolutely new kind of environment

### Training data
To obtain training data I used provided simulator. Driving 2 or 3 laps trying to stay ideally in the center.
After first experiments I also added to the train set some recovery maneuvers:

While recording is off, move to the road edge. Then enable recording and immediately return to the center of the road

Total images in train set: **3036**

![Training data example](trainset_example.jpg)

Above is an example of recovery as seen by central camera

### Validation data
Validation data is also generated with the simulator. But I was driving in reverse direction.
If I were driving in the same direction - I would have a high chance to obtain very similar images in train and validation set. 
This could lead to overfitting in validation set.

No recovery maneuvers were executed in validation set. Only riding in the center

Total images in validation set: **6045**
Validation set is made larger to be able to better control changes in validation accuracy.

![Validation data example](validation_example.jpg)

### Data augmentation
I have used only one augmentation technique: converting image to HSV colorspace. Mostly because of V channel. Since it is more robust to changes in the lighting conditions.

_Please note_: [drive.py](drive.py) was modified to automatically augment images.

## Training
So far I have tried 2 network architectures. 
First is AlexNet-like network and second is VGG-like. 
Both networks are not "100% as described in original papers". They rather look like AlexNet and VGG. But for simplicity I will refer to them as AlexNet and VGG further.

To find best network architecture and hyperparameters I used grid search.
With grid search I optimized:
* learning rate
* network architecture
* dropout
* use of batch normalization
* dataset augmentation method

I am not using **fit_generator** because on my machine all images can freely fit to memory. And it is much faster than reload them on each epoch

## Anti-overfitting measures
Dropout and BatchNormalization are used to reduce overfitting. 
BatchNormalization also helps to speed up covergence.

## Final network description
AlexNet and VGG showed comparable results on validation dataset. But AlexNet runs up to 3 times faster. And model performance is very important in such real-time task as car driving. So my final net is AlexNet

With grid I have found that best results are achieved when BatchNormalization os turned on and dropout is set to 0.7

My AlexNet consists of 4 Convolution layers, each followed by relu, BN and max pool layers.
After that go two hidden layers with relu, batch normalization and dropout.
And final layer is a single value - predicted steering angle.

See [model.png](model.png) for full network graph (it is too big to include in report)

I use Adam optimizer and **mean square error** as loss function

## Conclusion
Final model is able to drive test track almost perfectly - it never hits yellow lines. However it fails to drive the second track at all.
There are only two explanations for this: 
1. model has "learned"(overfitted) first track and can't drive anywhere else.
2. training only on the first track is just not enough to drive on the second.

I have also noticed that model performance depends much more on what I show to the network, rather than which network architecture/hyperparameters I use. 

I am going to address above issues in the next iteration of this project.
