
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 3 - Evaluation
# 
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[1]:

import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[2]:

def answer_one():
    
    dataset = pd.read_csv('fraud_data.csv')

    #dataset.head()

    count = dataset['Class'].value_counts()

#count
#count[0],count[1]
#count[1]/count[0]
    
    return count[1]/count[0]

answer_one()


# In[3]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y_test.shape


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[4]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score

    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)

    #y_dummy_predictions

    accuracy_ = dummy_majority.score(X_test,y_test)

    recall_ = recall_score(y_test,y_dummy_predictions)

    #accuracy_,recall_

    return accuracy_,recall_

answer_two()


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[5]:

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    SVC_data = SVC().fit(X_train, y_train)

    y_SVC_predictions = SVC_data.predict(X_test)


    accuracy_ = SVC_data.score(X_test,y_test)
    recall_ = recall_score(y_test,y_SVC_predictions)
    precision_ = precision_score(y_test,y_SVC_predictions)

    accuracy_,recall_,precision_

    return accuracy_,recall_,precision_

answer_three()


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[6]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    SVC_cust = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)

    y_SVC_cust = SVC_cust.decision_function(X_test)

    #y_SVC_cust = SVC_cust.predict(X_test)
    #y_SVC_cust

    y_pred = y_SVC_cust>=-220
    #y_SVC_cust.shape
    #y_pred

    y_pred = y_pred.astype(int)

    confusion_mc = confusion_matrix(y_test, y_pred)
    confusion_mc

    return confusion_mc

answer_four()


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[7]:

def answer_five():
        
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    #import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    lr = LogisticRegression().fit(X_train, y_train)
    y_proba_lr = lr.predict_proba(X_test)

    #y_proba_lr
    #y_proba_lr[:,1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lr[:,1])
    closest_thresh = np.argmin(np.abs(precision-0.75))
    closest_thresh_r = recall[closest_thresh]

    #plt.figure()
    #plt.xlim([0.0, 1.01])
    #plt.ylim([0.0, 1.01])
    #plt.plot(precision, recall, label='Precision-Recall Curve')
    #plt.show()

    #precision
    #closest_thresh_r

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr[:,1])


    #plt.figure()
    #plt.xlim([-0.01, 1.00])
    #plt.ylim([-0.01, 1.01])
    #plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve')
    #plt.xlabel('False Positive Rate', fontsize=16)
    #plt.ylabel('True Positive Rate', fontsize=16)
    #plt.show()

    closest_tpr = np.argmin(np.abs(fpr_lr-0.16))
    closest_tpr_val = tpr_lr[closest_tpr]

    #closest_tpr_val

    return closest_thresh_r,closest_tpr_val

answer_five()


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# In[8]:

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression


    grid_values = {'C': [0.01, 0.1, 1, 10, 100]}

    A=[]

    for g in ['l1','l2']:
        lr = LogisticRegression(penalty=g).fit(X_train, y_train)
        grid_lr_rec = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
        grid_lr_rec.fit(X_train, y_train)

        grid_lr_rec.cv_results_['mean_test_score']
        A.append(grid_lr_rec.cv_results_['mean_test_score'])

    #A
    #type(A)
    B=np.asarray(A)
    #B

    return np.transpose(B)

answer_six()


# In[9]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())


# In[ ]:



