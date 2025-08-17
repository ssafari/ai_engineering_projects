# AI Machine Learning Projects
Applied Data Science Program: Leveraging AI for Effective Decision-Making

2 projects are presented in this repository, part of the learning Applied Data Science Program.
Both projects are linear regression and supervised learning predciting continues values.

### Capston Project: Used Car Price Prediction Model
Creating an ML Model for used car market to predict the used car prices.
### Elective Project: Boston Housing Price Prediction Project
Creating an ML Model for predicting house prices using training dataset.
The final trained model will be use to predict the parice.



# Summary

In a linear regression model preparation, the key for finding the best possible model is reducing the error between the predicted values and the actual values. 
This is done through a loss function calculation of Mean Squared Error (MSE). The goal in these project is to minimize MSE, which means reducing the average 
squared error across all data points. The smaller the MSE, the better the model’s predictions are. This is where we optimize linear equation parameters by
using the most common optimization technique the gradient descent.  After multiple iterations of gradient descent, the final model converges, meaning the 
parameters no longer change significantly, and the MSE is minimized.

## Data Analysis Techniques

For large dataset the common approche is to start by understanding and preparing data, such as the type of features, their correlation, and the target variable, 
for performing feature selection. In other words making data clean and suitable for ML algorithm to use it.  Once data is ready we can start our model building 
by splitting data and measuring model performance, until we get the best possible performance.

**Steps used in these projects to prepare data:**

- Data preparation
  - Understanding predicted values and independent values 
  - Remving duplicate values
  - Handling null values in the dataset rows.
  - handling categorical data
  - removing non signigicants data columns which they are not influencing predicted values
- EDA:
  - Univariate data analysis
  - Bevariate data analysis
  - Correlation analysis and finding feature importance using heatmap
  - Feature engineering

### Model Performance Techniques

Mode performace can be affected by overfitting or underfitting data. If the model overfits, then it will perform well on the training data but won't be able to reproduce 
the results on the validation data, and if it underfits, then it will perform poorly on both the training and validation datasets. This creates the possibility of an exchange 
or tradeoff between bias and variance.  To overcome this problem one way to to split our dataset to large training dataset. Another technique to overcome this issue it to use
regularization L1, L2 techniques.  Use cross-validation techniques to evaluate model performance on unseen data and identify potential overfitting.

- cross-validation methods
  - Hold-out cross-validation (splitting data to):
    - training data
    - testing data
  - K-Fold cross-validation (not used in these projects)
  - LOOCV cross-validation (not used in these projects)
- Creating the model and fitting it with splited data
- Regularizations: (Bias Variance Trade Off - Overfitting and Underfitting)
  - L1: LASSO - for effectively performing feature selection by removing irrelevant features from the mode in high-dimentional data
  - L2: Ridge Regression - to prevent overfitting and improve model stability.
- Decision Tree
- Random Forest
- Statical Analysis

### Regression Model Assemption

- **Linearity** - linear relation between dependent and independent variables.
- **No Multicollinearity** - it means no correlation between independent variables, hence, it is necessary to include only non-correlated independent variables in the model. To detect that we can use VIF method to calculate how much the variance of regression coefecient is inflated.
- **Homoscedasticity** - the error associated with each data point should be equally spread (meaning “constant variance”) along the best fit line.
- **No Endogeneity** - it is assumed that the independent variables being correlated to the error terms of the model.

### Cost function types in linear regression:
Evaluate the model with evaluation metrics

- ***Mean Square Error (MSE)*** - the differences between predicted and actual values, giving more weight to large errors. A smaller MSE value indicates a better model fit.
- ***Mean Absolute Error (MAE)*** - useful when dealing with outliers or when we want a simpler interpretation of error.
- ***Root Mean Squared Error (RMSE)*** - measure of error that maintains sensitivity to larger mistakes making it suitable for many regression tasks. Because the error term is squared, it is more sensitive to outliers.
- ***R-squared (Coefficient of Determination)*** - It is a widely used metric for evaluating the explanatory power of a linear regression model. It is useful for linear regression models where the relationship between the variables is non-linear. It ranges between 0 and 1, closer to 1 indicates a perfect fit.

### Most used Python libraries:

- **numpy**: For handling arrays.
- **pandas**: For handling and loading large datasets.
- **matplotlib.pyplot**: To create the scatter plot for visualization.
- **seaborn**: (optional) For better aesthetics.
- **train_test_split**: To split the dataset into training and testing sets.
- **StandardScaler**: For scaling features if necessary (optional at this stage).







