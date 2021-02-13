# Chest-Xray-prediction
Predict X-rays with pneumonia using classification models.

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If you do not have Python installed yet, it is highly recommended to install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has some of the above packages and more included. 

### Dataset

The dataset and more informations about it can be found in the following [link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Code

There are three different notebooks

- ExtractFeatures.ipynb, which reads the images and extract the features into pickle files.
- MainNotebook.ipynb, which contains the evaluation of the models.
- CompareImageKeypoints.ipynb, which compare keypoints of two images, its only about to help the reader understand what keypoints are.

#### ExtractFeatures

Location of the dataset needs to be set.
Transform images to arrays, by extracting their features.
Currently implemented 3 different types of feature extraction.

- Image features
- KAZE features
- HOG features

For more information check the notebook.

[`view notebook`](https://nbviewer.jupyter.org/github/teoad95/Chest-Xray-prediction/blob/main/ExtractFeatures.ipynb)

#### MainNotebook

Contains models training, evaluation and tuning.
The following models used:

- Logistic regression
- KNN
- Discriminant analysis
- Gaussian naive bayes
- SVM

Models compared using:

  - Their accuracy on validation set.
  - Confusion matrix of testing set.
  - Precision score of testing set.


For more information check the notebook.


