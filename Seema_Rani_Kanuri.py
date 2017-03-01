
# coding: utf-8

# In[32]:

#Importing the packages and library required
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import ensemble


# In[33]:

#Load train and test csv file
train_df = pd.read_csv('G:/OneDrive - Texas Tech University/Masters/DS- Multivariate Analysis/Titanic/train.csv')
test_df = pd.read_csv('G:/OneDrive - Texas Tech University/Masters/DS- Multivariate Analysis/Titanic/test.csv')
test_df1 = pd.read_csv('G:/OneDrive - Texas Tech University/Masters/DS- Multivariate Analysis/Titanic/test.csv')


# In[34]:

#Display the test data
test_df.head()


# In[35]:

#Finding the number of NUll or NaN values in the dataframe
test_df.isnull().sum()


# In[36]:

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)


# In[37]:

test_df[['Sex', 'Pclass','Fare','Age', 'Name']]


# In[38]:

#Display the train data
train_df.head()


# In[39]:

#Finding the number of NUll or NaN values in the dataframe
train_df.isnull().sum()


# In[40]:

#The fare $79.65 of PassengerID=559 for a Pclass=1 passenger departing from Embarked 'S' coincides nicely with the Fare of $80 paid by passengerID=62.
#Imputting the NaN of Embarked for passengerId=62 values with 'S'.


train_df.loc[train_df['Embarked'].isin(['S','Q','C']) == False]
train_df.loc[(train_df['Fare'] >= 79) & (train_df['Fare'] <= 81) & (train_df['Age'] <= 39) & (train_df['Age'] >= 37)]
train_df.loc[(train_df['PassengerId'] == 62,'Embarked')]= "S"


# In[41]:

#The fare $62.0 of PassengerID=588 for a Pclass=1 passenger departing from Embarked 'C' coincides nicely with the Fare of $80 paid by passengerID=830.
#Imputting the NaN of Embarked for passengerId=830 values with 'C'.

train_df.loc[(train_df['Fare'] >= 79) & (train_df['Fare'] <= 81) & (train_df['Age'] <= 63) & (train_df['Age'] >= 60)]
train_df.loc[(train_df['PassengerId'] == 830,'Embarked')]= "C"


# In[42]:

train_df['Sex'] = pd.factorize(train_df['Sex'])[0]
test_df['Sex'] = pd.factorize(test_df['Sex'])[0]


# In[43]:

train_df.Embarked.replace(to_replace='S', value=0, inplace=True)
train_df.Embarked.replace(to_replace='Q', value=1, inplace=True)
train_df.Embarked.replace(to_replace='C', value=2, inplace=True)
test_df.Embarked.replace(to_replace='S', value=0, inplace=True)
test_df.Embarked.replace(to_replace='Q', value=1, inplace=True)
test_df.Embarked.replace(to_replace='C', value=2, inplace=True)


# In[44]:

train_df['Family']= train_df['SibSp'] + train_df['Parch']
test_df['Family']= test_df['SibSp'] + test_df['Parch']


# In[45]:

train_df.Family.unique()


# In[46]:

test_df.Family.unique()


# In[47]:

train_df.loc[(train_df['Family'] > 0, 'Family')]= 1
train_df.loc[(train_df['Family'] == 0, 'Family')]= 0
test_df.loc[(test_df['Family'] > 0, 'Family')]= 1
test_df.loc[(test_df['Family'] == 0, 'Family')]= 0


# In[48]:

test_df.drop(['Cabin','SibSp','Parch','Name','Age','PassengerId','Ticket'],inplace=True,axis=1)


# In[49]:

train_df.drop(['Cabin','SibSp','Parch','Name','Age','PassengerId','Ticket'],inplace=True,axis=1)


# In[50]:

train_df


# In[51]:

X_train = train_df.iloc[:,1:]
Y_train = train_df.iloc[:,0]


# In[52]:

#Model : Random Forests
random_forest = RandomForestClassifier(n_estimators = 1000)
random_model = random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)

Y_pred_random_forest = random_forest.predict(test_df)


# In[53]:

#Model : Logistic Regression
logreg = LogisticRegression()
Logistic_model = logreg.fit(X_train, Y_train)
logreg.score(X_train, Y_train)

Y_pred_logreg = logreg.predict(test_df)


# In[54]:

#Model  : Support Vector Machines

svc = SVC()
SVC_model = svc.fit(X_train, Y_train)
Y_pred_SVC = svc.predict(test_df)
svc.score(X_train, Y_train)


# In[55]:

output = pd.DataFrame({'PassengerId': test_df1['PassengerId'],'Survived': Y_pred_random_forest})
output.to_csv('C:/Users/Seema Kanuri/Desktop/SeemaPredOutput_RandomForest.csv', index=False)

output = pd.DataFrame({'PassengerId': test_df1['PassengerId'],'Survived': Y_pred_logreg})
output.to_csv('C:/Users/Seema Kanuri/Desktop/SeemaPredOutput_Logistic.csv', index=False)


output = pd.DataFrame({'PassengerId': test_df1['PassengerId'],'Survived': Y_pred_SVC})
output.to_csv('C:/Users/Seema Kanuri/Desktop/SeemaPredOutput_SVC.csv', index=False)

