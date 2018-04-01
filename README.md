# **Traffic Sign Recognition** 

## 1. Problem Statement

In this exercise, a supervised deep learning approach is applied for German traffic sign classification. Input data are small color images (32x32x3) for traffic sign, and output is its corresponding labels (43 classes).

A classic data separation of train/dev/test is used, and some new test images from web are used to test generalization of the model.

## 2. Training Strategy
(1) Data preparation:
    - Whiten input data to zero mean
    - Apply data augmentation
(2) Network design:
    - Decide feature abstraction level according to problem complexity
    - Decide network layout based on image characteristics
(3) Regularization techniques:
    - Apply dropout between FC layers
    - Apply L1 and L2 kernel regularizers
(4) Optimization techniques:
    - Choose approporiate optimizer and training hyperparameters
    - Apply batch-normalization and learning rate decay for better convergence
(5) Training evaluation
    - Evaluate overfitting on train and dev set
    - Test on new images

These steps are to be discussed in details in the following.

## 3. Data Preparation

### 3.1 Dataset information

The training set contains 34,799 images, validation set contains 4,410 images, and test set contains 12,630 images.
The shape of all images are of 32x32x3, and there are 43 classes in total.
One training example is shown below, whose label index is 31 (refering to: Wild animals crossing)
[image1]: ./report_imgs/train_image_sample.png "training image sample"
![alt text][image1]

### 3.2 Data Preprocessing
Whitening input data to zero mean usually helps first layer's activation. In this exercise, 128 is removed for a 8-bit input image in each color channel. 

Data augumentation is applied as well to help the model to generalize. Traffic sign contains distinctive structure information, so usually excessive geometrical transform (such as flipping, mirroring, rotating, perspective or other affine transform) would not be appropriate. 

Therefore, in this excerise, only basic data augmentation such as cropping, blurring, additive noise, and intensity adjustment are added. Without data augumenation, training set overfitting cannot be easily avoided. 

In addition, trainning data is shuffled at each epoch to reduce the risk of model's memorizing the training set.

## 4. Network Design

### 4.1 Decide feature abstraction level according to problem complexity

It is generally beleived that along with the stacking of convolution layers, higher levels of feature extraction can be achieved. For example:
- First conv layer is responsible for primiary feature extraction, such as edges.
- Second conv layer is responsible for combinations of primary features, such as corners.
- Third conv layer makes structural descriptors by combining previous features, such as shape-context, or SIFT-type descriptor.
- Fourth conv layer continues to group features into component representations, such as bag-of-words.
- Beyond fifth layer, it would be hard to find intuitive correspondence. In general, even higher level abstraction is resulted.

However, as far as traffic sign is concerned, it seems due to the nature of design, only distinctive structrual information is used, so further stacking on conv layers would propbably create overwhelming description power so that the model might overfit the training set. Thus, in this exercise, only 3 layers of convolution are adopted, followed by fully-connected layers as classifier.

The table below shows the depth of model.

|     Index      |          Layer     		|        Purpose  	        | 
|:--------------:|:------------------------:|:-------------------------:| 
|       1        |      Convolution         |    Edge extraction        |
|       2        |      Convolution         |    Landmarker extraction  |
|       3  		 |	    Convolution         |    Shape descriptor    	|
|   4 & 5 & 6	 |    Fully connected       |    Classifier             |
 
### 4.2 Analyze image characteristics to setup network layout

After defining the depth of network, the width of network is defined based on the following input data analysis:
- For input image:
    - Color information would be essential since different color signs usually indicate different conditions. So color input format is kept.
    

- For convolution layer 1:
    - Simple structrial information is supposed to be the essence of traffic sign design, so 4 primary orientations for feature detection would be sufficient.
    - Since input image is small(32x32), a 5x5 kernel is used to collect local spatial information, and no stride would be necessary.
    - combining color channel number (3) and structural feature detector number (4), 12 hidden units (i.e. 3x4) are used in first conv layer to account for color edges.


- For convolution layer 2:
    - Smaller kernel is applied (3x3) due to reduced neighborhood.
    - Stacking 2 conv layers (3x3) would give an effective receptive field of (5x5), so no immediate pooling is effectuated. 
    - After conv1 (with pooling), data volum is: 14x14x12=2350, to keep similar data throughput, 16 hidden units are used which gives 12x12x16=2304.
    

- For convolution layer 3:
    - Used 24 hidden units to keep data throughput (10x10x24=2400)
    - Pooling for data reduction to avoid exploding connections in FC layer
   

- For fully connected layers
    - Applied 3 FC layers in order to have 2 chances of dropout insertion which helps to generalize model in test data
    - Gradually boil down data from 600 hidden units(after flattening) to 256, then to 128, then to 43 classes of traffic sign.
    
The table below shows the depth and width of model.

|     Index      |     Layer(depth)   		|      Depth Purpose  	    |   Activation(width)   |    Width Purpose        | 
|:--------------:|:------------------------:|:-------------------------:|:---------------------:|:-----------------------:| 
|                |        Input             |                           |    32 x 32 x 3 (3072) |                         |
|       1        |      Convolution         |    Edge extraction        |    14 x 14 x 12(2350) |  (5,5) for neighborhood |
|       2        |      Convolution         |    Landmarker extraction  |    12 x 12 x 16(2304) |    keep data throughput |
|       3  		 |	    Convolution         |    Shape descriptor    	|    10 x 10 x 24(2400) |    keep data throughput |  
|                |       Pooling            |                           |    5  x 5  x 24(600)  | dimensionality reduction|
|       4        |    Fully connected       |    Classifier             |    256 (reduce to 40%)| gradually boiling down  |
|       5        |    Fully connected       |                           |    128 (reduce to 50%)| gradually boiling down  |
|       6        |    Fully connected       |                           |    43  (reduce to 30%)| gradually boiling down  |


## 5. Regularization

Dropout helps the network not to create unique dependency on certain features, which enables better generalization performance. 

L2 regularization leads to weight decay, making the model less sensible to outliers.

L1 regularization creates sparse connections, which might be especially true in traffic sign design, whose structral features are so distinguish and exclusive to each other, i.e., not all features are active at the same time for a certain input, thus leading to sparse kernels.

In this exercise, L1 and L2 kernel regularizer are applied over all layers, and dropout layer is inserted between fully connected layers. Convolution layers are already small and compact which are not subject to huge data redundancy, so no convolutional layer are influnced by dropout. 

Dropout rate applied is 0.5, and regularizer parameter is set to 1.0 to both L1 and L2 regularization.

## 6. Optimization
Batch normalization is applied to help training converge, and it is systematically applied after each convolutional layer.

Relatively large mini-batch size is applied as well in order for the model to converge faster. 

Learning rate decay is applied after a certain epoch to better convergence. 

The initial learning rate is set to be 0.001, and it decays to 1/10 after each 100 epochs.

## 7. Training and Evaluating

Training process is performed for 300 epochs, each of which trains on training set, and evaluated on dev set. At the end of training, the performance of the trained model is evaluated on test data which are reserved aside and never used in training. The obtained prediction accuracy is around 97.4%.

When training and evaluating are finished, new test images are downloaded from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) for real world image test. 9 random images are drawn at each test from 1000 real world images.

The image below show one test run:
[image2]: ./report_imgs/test_new_images.png "New test images"
![alt text][image2]

For each test image, top 5 candidates are listed in the table below:

|  Image  |        1st 		 |        2nd  	      |       3rd         |        4th       |       5th        |
|:-------:|:----------------:|:------------------:|:-----------------:|:----------------:|:----------------:| 
|    1    |No passing(>3.5t) |  Speed(<80km/h)    |    Double curve   |  Speed(<100km/h) |   Priority road  |
|    2    |    No vehicles   |  Speed(<60km/h)    |    Yield          |  Speed(<120km/h) |    No passing    |
|    3    |    Keep right    |  Turn left ahead   |    End of speed   |  Roundabout      |straight or right |
|    4    | Speed(<30km/h)   |  Speed(<50km/h)    |    Speed(<70km/h) |  Speed(<20km/h)  | Speed(<80km/h)   |
|    5    | Speed(<100km/h)  |  Speed(<30km/h)    |    Roundabout     |  right-of-way    |   Priority road  |
|    6    | Traffic signals  |  Bumpy road        |  General caution  |  Dangerous curve |    Road narrows  |
|    7    |    Ahead only    |  Turn left ahead   | straight or right |  No passing      | Speed(<60km/h)   |
|    8    | Speed(<30km/h)   |  Speed(<70km/h)    |    Speed(<50km/h) |  Speed(<20km/h)  | Speed(<80km/h)   |
|    9    | Speed(<60km/h)   |  Speed(<80km/h)    | Bicycles crossing |  Slippery road   | Speed(<50km/h)   |

## 8. (Optional) Visualizing Activation Map
The convolutional kernels themselves are too small to give intuitive information. However, it is possible to visualize the activation map to check which part of images has most actively contributed to pattern recognition. 

For the following test sample, the activation map of convolutional layers are shown below:

[image3]: ./report_imgs/test_image_sample.png "Test image sample"
![alt text][image3]

[image4]: ./report_imgs/activation_map_conv1.png "Activation map(conv1)"
![alt text][image4]

[image5]: ./report_imgs/activation_map_conv2.png "Activation map(conv2)"
![alt text][image5]

[image6]: ./report_imgs/activation_map_conv3.png "Activation map(conv3)"
![alt text][image6]