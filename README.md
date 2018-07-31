# MONET

## Data
Follow the intructions in this website http://domedb.perception.cs.cmu.edu/171026_pose3.html to download the images. The data is divided into labeled and unlabeled data which are stored in label.txt and unlabel.txt. We put some images in the image directory to show how the images should be organized.

## Train a base model
Run cpm_training_data.py to generate tfrecord training data using label.txt.

Run cpm_train.py to train convolutional pose machine model for labeled data.


## Epipolar supervision
Run Epi_LabeledData.py to generate training data for labeled data.

Run Epi_UnlabeledData.py to generate training data for unlabeled data and their pairs.

Run Epi_Semisupervised.py to train the base model with epipolar supervision.
