# Classification with Conventional Neural Network Using Inception V3 Architecture for Skin Cancer Diagnosis Detection

&nbsp;
## Introduction
Here is a mini project that we propose to fulfill the Final Semester Examination (UAS) even in semester 6. This assignment is intended to fulfill the course of Image Processing and Computer vision. 
Our proposal is entitled "**Classification with Conventional Neural Network Using Inception V3 Architecture for Skin Cancer Diagnosis Detection**". 

In this implementation, we will detect skin cancer diagnoses with the type of cancer in the form of images which will be produced in the form of skin cancer diagnoses with 2 labels, namely Malignant and Benign. Malignant label indicates that the cancer is detected malignant skin cancer and for Benign label means that the cancer is detected benign skin cancer.  This research was conducted using the Convolutional Neural Network (CNN) method using InceptionV3 architecture.

Classification that is done using a dataset obtained from the International Skin Imaging Collaboration (ISIC) dataset on the uploaded kaggle that has been uploaded to S3 on AWS with number of categories as many as 3 classes namely melanoma, nevus, and seborrheic keratosis. 

<p align="center">
 <img align="middle"  alt="Sampel Data" width="80%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Sampel%20Data.png"/>
</p>

&nbsp;
## Dataset
Download the dataset and unzip it at the following link :
#### 1. Download and unzip the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB).
#### 2. Download and unzip the [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (824.5 MB).
#### 3. Download and unzip the [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB).

&nbsp;
## Preprocessing and Labeling Result
The following are the results of preprocessing image data that has been encoded into a 3D tensor with uint8 data type with color channel
RGB COLOR CHANNEL. Next, change the data type to float32 with the range [0,1], then resize the image to size [299, 299] and the image is returned as a resized tensor. The next step is to prepare the training and validation datasets with the appropriate batch size and chaching for the model training process. The following image shows one batch of
validation dataset with the appropriate labels. 
<p align="center">
 <img align="middle"  alt="Hasil Label" width="80%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Hasil%20Label.png"/>
</p>

&nbsp;
## Model used and generated

<p align="center">
 <img align="middle"  alt="Hasil Label" width="80%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/InceptionV3.png"/>
</p>

The following are the stages of building the [InceptionV3](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4) architecture model. 

The model consists of two parts: the [InceptionV3](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4) feature layer and the Dense layer for output. This layer uses module_url to load the pre-trained [InceptionV3](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4) architecture as the feature layer. The Dense layer with one unit and a sigmoid activation function. This layer is responsible for generating the prediction output in the form of a number between 0 and 1, which represents the probability of a positive class.

Compile the model by determining the loss function to be used, `binary_crossentropy` is used in binary classification tasks and serves to measure the error or difference between the model's predicted value and the target value in binary classification tasks in the hope of getting more predictions. The optimizer used in model training is Root Mean Square Propagation `RMSprop` which is an optimization algorithm in deep learning models because of its effectiveness in overcoming large gradient problems and accelerating model convergence.

&nbsp;
## Classification Results 
### Confusion Matrix
<p align="center">
 <img align="middle"  alt="Hasil Label" width="40%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Confusion%20Matrix.png"/>
</p>

**Keterangan**
 - True Positive (TP) : 0.74
 - False Positive (FP) : 0.39
 - True Negative (TN) : 0.62
 - False Negative (FN) : 0.26

### Sensitivity & Specificity
<p align="center">
 <img align="middle"  alt="Hasil Label" width="30%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Sensitivity%20dan%20Specificity.png"/>
</p>
Sensitivity is a measure of the extent to which the model can correctly identify positive (malignant) cases out of all cases that are actually positive. The result of sensitivity is 0.735 which means the model has the ability of 73.5% to correctly identify skin cancer cases that are actually positive (malignant). Specificity is a measure of the extent to which the model can correctly identify negative (benign) cases from all cases that are actually negative. The result of specificity is 0.617 which means the model has the ability of 61.7% to correctly identify skin cancer cases that are actually negative (benign).

### ROC Curves
<p align="center">
 <img align="middle"  alt="Hasil Label" width="50%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/ROC%20AUC.png"/>
</p>

The ROC AUC (Area Under the Curve) value is 0.676. ROC AUC is a measure that describes the extent to which the model can distinguish between positive and negative classes. In this context, the ROC AUC value of 0.676 indicates that the model performs reasonably well in distinguishing between positive (malignant) skin cancer cases and negative (benign) skin cancer cases.

&nbsp;
## Test Data Prediction Results and Accuracy
### Prediction 1: Class `Malanoma` Result `Benign` with accuracy `95.39%`
<p align="center">
 <img align="middle"  alt="Hasil Label" width="50%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Prediksi%201.png"/>
</p>

### Prediction 2 : Class `Malanoma` Result `Benign` with accuracy `97.35%`
<p align="center">
 <img align="middle"  alt="Hasil Label" width="40%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Prediksi%202.png"/>
</p>

### Prediction 3 : Class `Malanoma` Result `Malignant` with accuracy `64.27%`
<p align="center">
 <img align="middle"  alt="Hasil Label" width="50%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Prediksi%203.png"/>
</p>


### Prediction 4 : Class `Nevus` Result `Benign` with accuracy `95.56%`
<p align="center">
 <img align="middle"  alt="Hasil Label" width="50%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Prediksi%204.png"/>
</p>

### Prediction 5 : Class `Seborrheic Keratosis` Result `Benign` with accuracy `92.47%`
<p align="center">
 <img align="middle"  alt="Hasil Label" width="50%" src="https://github.com/sitiaisyah14/Skin_Cancer_Detection_with__TensorFlow_and_Keras_in_Python/blob/main/Image/Prediksi%205.png"/>
</p>

