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

### Validation data
Validation data is also generated with the simulator. But I was driving in reverse direction.
If I were driving in the same direction - I would have a high chance to obtain very similar images in train and validation set. 
This could lead to overfitting in validation set.

No recovery maneuvers were executed in validation set. Only riding in the center

Total images in validation set: **6045**
Validation set is made larger to be able to better control changes in validation accuracy.

## Training
So far I have tried 2 network architectures. 
First is AlexNet-like network and second is VGG-like. 

To find best network architecture and hyperparameters I used grid search.
With grid search I optimized:
* learning rate
* network architecture
* dropout
* use of batch normalization

## Anti-overfitting measures
I used dropout and BatchNormalization combined. 

## Final network description


