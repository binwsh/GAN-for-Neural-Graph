# GAN_for_Neural_Image
We implemented a combined GAN model to predict Autism Spectrum Disorder(ASD) and achieved about 3% increase in accuracy when compared with the previous best model.

## Dependencies
  * Keras
  * Pandas
  * Numpy
  * Sklearn

## Workflow
### 1. Preparation 
Convert ABIDE_fc.mat to csv files so that data become easily readable to Python. <br>
Create a directory named "FC_norm". Then, in MatLab, run
```
converter.m
```

### 2. Run ipynb files
The ipynb files in the folders named "ACGAN" and "BrainNet" preprocess and create datasets, define models, train and save models and evaluate models. 
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

