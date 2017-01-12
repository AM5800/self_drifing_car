# Project report: Behavioral cloning
## Project structure
* [model.py](model.py) - contains NN definition and functions to train it
* [drive.py](drive.py) - controls simulation
* [dataset.py](dataset.py) - utiliy functions for dataset loading and transformation
* [model.json](model.json) - final model structure definition
* [model.h5](model.h5) - final model's weights
* [model.cfg](model.cfg) - stores dataset name which was used during training

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

## Training
So far I have tried 2 network architectures. 
First is AlexNet-like network and second is VGG-like. 
Both networks are not "100% as described in original papers". They are rather look like AlexNet and VGG. But for simplicity I will refer to them as AlexNet and VGG further.

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
AlexNet and VGG showed comparable results on validation dataset. But AlexNet runs up to 3 times faster. And model performance is very important in such real-time task as car driving. So I am using AlexNet further.

With grid I have found that best results are achieved when BatchNormalization os turned on and dropout is set to 0.7

AlexNet consists of 4 Convolution layers, each followed by BN, relu and max pool layers.
After that goes one hidden layer with batch normalization, relu and dropout.
And final layer is a single value - predicted steering angle.

I use Adam optimizer and **mean square error** as loss function

## Conclusion

