# GAN_for_Neural_Image
We implemented a combined GAN model to imporve brain dysfunction prediction. And we applied the model to ABIDE data set and ADHD data set.

## Dependencies
  * Keras
  * Pandas
  * Numpy
  * Sklearn

## Workflow
### 1. Preparation 
####1. ABIDE_data
Download ABIDE_fc.mat from
https://uab.app.box.com/s/jaunccb8uo7hu4xxlf81iacvl3jildip
Put the mat file in the folder of Dataset.
Convert ABIDE_fc.mat to csv files so that data become easily readable to Python. <br>
Create a directory named "FC_norm". Then, in MatLab, run
```
converter.m
```

####2. ADHD_data
Preprocessing the data, run
```
Preprocessing.ipynb
```

### 2. Run ipynb files
The ipynb files in the folders named "ACGAN" and "BrainNet" preprocess and create datasets, define models, train and save models, evaluate models. 
EX: Open folder "ACGAN" and run
```
AC_mode0.ipynb
```

## Note
There are four modes (meaning different inputs) in one type of model:
1. Mode0: label + age + gender
2. Mode1: label + age
3. Mode2: label + gender
4. Mode3: label	

