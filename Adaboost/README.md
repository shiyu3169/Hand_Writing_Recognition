AdaBoost
========================

This folder contains the code of Adaboost for Handwriting digit recongization

The pre step is to downlowd the dataset from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Step 1
extract 200 features from dataset.

`ImageFeatureExtraction.java` - main class for image extraction

`HAARFeatureExtraction.java` - HAAR class, implement HAAR algorithm

## Step 2
Run adaboost to build model to predict and to get accuracy.

`Main.java` - main class

`ECOC.java` - implement ECOC class

`ECOCStump.java` - implement Adaboost

`DataInput.java` - helper class to read file
