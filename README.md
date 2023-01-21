Predicting Employee Attrition

This project aims to predict the possibility of an employee leaving the company.

DESCRIPTION

The data is for a company that is trying to control attrition. There are two sets of Data, 'existing employees' and 'employees who have left'. 
The following attributes are available for every employee:

 * Satisfaction level
 * Last evaluation
 * Number of project
 * Average monthly hours
 * Time spent at the company
 * Whether they have had a work accident
 * Whethere they have had promotion in the last 5 years
 * Departments (column sales)
 * Salary
 * Whether the employee has left
 
 Objective
 * What type of employees are leaving
 * Determne which employees are prone to leave next
 * Present your result in the presentation sheet presentation area
 
 APPROACHING THE PROBLEM
 
 In preparing the data set, I filled the 'have left' column in the two data sets by 0 and 1, where 'have left' in the Existing Emoployees data set is labelled as 0 while
 the 'have left' column in the Departed Employees data set is labelled 1. the 2 data sets where then merged. The percentage analysis of the exising and 
 departed employees and it shows that the data is imbalanced with the departed employees at 23.8% while that of existing of employees is at 76%, this shows 
 the data is imbalanced. Feature selection was done on the data to remove the colunms that do not have strong correlation with the target column. 
 
 libraries used for the project
 
 # Import Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_theme(style='ticks')

# Import ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV

#from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score


