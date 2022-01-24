# decision-tree

%--------------------------------------- Data -----------------------------------------%

The Pima Indian Diabetes dataset found Pima-Data-Adjusted.mat is from the UC Irvine Machine Learning Database at https://archive.ics.uci.edu/ml/datasets.php. 
The first 8 columns are the features and the last column represents the class label ( diabetes = , not diabetic= ). 

Note that we adjusted the data from the database to remove entries with missing data, and these data have been normalized. 

%--------------------------------------- Goal for Project -----------------------------------------%

We use 5-fold cross validation to tune hyperparameters, e.g., by using a 70/30 train/val split.

## Model 1: A decision tree

In this model we train a decision tree to determine if patients are diabetic or not. 

We tune the hyperparameters: maximum depth (max\_depth), minimum node size (min\_samples\_split), gain in error reduction. 

## Model 2: A random forest

We use the optimal hyperparameters from Model 1 to define the trees.

We tune the hyperparameter ``n_estimators'', i.e., the number of trees. 



%--------------------------------------- Required Packages -----------------------------------------%
