# AI-Machine Learning Projects
Applied Data Science Program: Leveraging AI for Effective Decision-Making

2 projects are presented in this repository, part of the learning Aplied Data Science Program:

### Capston Project: Used Car Price Prediction Model
Creating an ML Model for used car market to predict the used car prices.
### Elective Project: Boston Housing Price Prediction Project
Creating an ML Model for predicting house prices using training dataset.
The final trained model will be use to predict the parice.



##Summary
In a linear regression model preparation, the key for finding the best possible model is reducing the error between the predicted values and the actual values. 
This is done through a loss function calculation of Mean Squared Error (MSE). The goal in these project is to minimize MSE, which means reducing the average 
squared error across all data points. The smaller the MSE, the better the model’s predictions are. This is where we optimize linear equation parameters by
using the most common optimization technique the gradient descent.  After multiple iterations of gradient descent, the final model converges, meaning the 
parameters no longer change significantly, and the MSE is minimized.

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



Techniques used to solve that problem are:

- Data analysis
- EDA:
  - Univariate data analysis
  - Bevariate data analysis
  - Correlation analysis using heatmap
  - Feature engineering
- Data splitting to:
  - training data
  - testing data
- Linear Regession
- Ridge Regression
- Decision Tree
- Random Forest
- Statical Analysis




