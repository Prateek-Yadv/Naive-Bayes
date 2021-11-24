#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
salary = pd.read_csv('C:/Users/prate/Downloads/Assignment/Naive Bayes/SalaryData_Train.csv')


# In[9]:


salary.head()


# In[10]:


salary.isnull().sum()


# In[11]:


salary.info()


# In[13]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #training and testing data split
salary= salary.apply(LabelEncoder().fit_transform)
salary.head()


# In[14]:


Y = salary["Salary"]
X = salary.iloc[:,:-1]
X.head()


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[18]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[ ]:





# In[19]:


# Preparing a naive bayes model on training data set 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.naive_bayes import GaussianNB

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_train, Y_train)
 


# In[21]:


cross_val_score(pipeline, X_train, Y_train, scoring='roc_auc', cv=10).mean()


# In[40]:


accuracy_test_g


# In[22]:


Y_pred=pipeline.predict(X_test)


# In[23]:


acc = accuracy_score(Y_test, Y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y_test, Y_pred)

