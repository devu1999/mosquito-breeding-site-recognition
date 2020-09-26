## DIGITAL IMAGE PROCESSING PROJECT 
IMAGE CLASSIFICATION FOR MOSQUITO BREEDING GROUND

## Abstract

Mosquitoes are one of the largest causes behind deadly diseases in the world. Identifying their breeding grounds can help in preventing such diseases. In Indian scenario where sanitation is still a major issue, this identification becomes all the more important.Dengue and malaria are on rise throughout the country. Even metro cities have plenty of stagnant water puddles which help in breeding of mosquitoes. Identifying such water puddles will eliminate the root cause of these diseases.

## Problem statement

* The crux of the problem lies in determining stagnant water in the images in the form of puddles or any other water formations.

* The probability of the region depicted in the image being a breeding ground has to be calculated by running the image through  a series of trained algorithms.

* This boils down to an image processing problem which involve feature extraction and processing raw data.


## Team members
- Ayush Shivani
- Devansh Gupta
- Shubhankar Bhagwat

## Folder contents
- /data : images collected from google search
- /data/dataset_with_rotated_images: dataset with rotated images as well

## Specific objectives

The goal of the project is to train a water buddle detection system. The project consists of:

1. Collecting images from google search and other online mediums.
2. Rotate the images for better inclusivity in the dataset.
3. Use image processing techniques like SIFT for feature extraction.
4. Choose a classification model and configuration
5. Train the model and vary the parameters accordingly for better results.
6. Validate the model and test the results.

## Details for each step

1. Our main source for the images in the dataset is google search. We used various tags to collect around 211 images which have water puddles in it. however, it became a challenge to collect images without puddle in them but with the same environment background as that of images with puddle. After searching meticulously, 160 images were collected which didn't have puddles in it.

2. As implemented in the paper, we also use rotated images to provide the necessary directional variance in the dataset. [rotated_images.ipynb](https://github.com/ayushshivani/dip_project/blob/master/rotate_imges.ipynb) is used to get the rotated images. The rotation angle is randomly chosen between 0 to 60. 
We made two more rotated images for a image in the dataset. Thus, we have 633 images with puddles and around 480 images without puddle.

3. This step is about pre-processing the images before choosing a model and training it on our image dataset for classification. We use SIFT (Scale-Invariant Feature Transform) for feature extraction in the images which are then used directly for training the model.

4. In the search of the most optimal classifier for our problem statement, we found the following classifiers are giving better results as compared to others:

- Support vector machines(SVMs)
- Logistic regression 
- Random forest classifier
- Bayesian classifier
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

in order to use the benefits of all these classifier, we use the majority voting over them to find the final optimal result.

5. The images in the [dataset_with_rotated_images](https://github.com/ayushshivani/dip_project/tree/master/dataset_with_rotated_images) are first splitted into training and testing data with 80-20 distribution. The training data is then used to train the above model with majority voting over multiple classifiers.

6. The testing data is then used to validate the trained model.

## Project pipeline


![Project pipeline image](https://github.com/ayushshivani/dip_project/blob/master/pipeline.png)

## How to run the code 
1. pip3 install -r requirements.txt
2. python3 final_code.py

### How to run Demo
1. cd demo
2. pip3 install -r requirements_demo.txt
3. python3 app.py
4. Visit localhost:5000 in the browser
5. Upload the image to be classified.



