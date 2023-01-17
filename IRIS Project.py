#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print("sys version:{}".format(sys.version))
import scipy
print("scipy version:{}".format(scipy.__version__))
import numpy
print("numpy version:{}".format(numpy.__version__))
import matplotlib
print("matplot version:{}".format(numpy.__version__))
import sklearn
print("sklearn version:{}".format(sklearn.__version__))
import pandas
print("pandas version:{}".format(pandas.__version__))

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','peta-width', 'class']
dataset = pandas.read_csv(url,names=names)


# In[3]:


print(dataset.shape)


# In[4]:


print(dataset.head(10))


# In[5]:


print(dataset.describe())


# In[6]:


print(dataset.groupby('class').size())


# In[7]:


dataset.plot(kind = 'box',subplots = True, layout = (2,2), sharex = False, sharey = False)


# In[8]:


dataset.hist()


# In[9]:


scatter_matrix(dataset)


# In[10]:


array = dataset.values
X = array[:,0:4]#all the columns from row 0 to 4
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[11]:


seed = 6
scoring = 'accuracy'


# In[12]:


#Sport Check Algorithm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Evaluate each Model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed, shuffle = True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    


# In[ ]:





# In[ ]:




