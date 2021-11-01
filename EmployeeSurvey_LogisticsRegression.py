#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##CHIKA IBEGBU PROJECT ON EMPLOYEE SURVEY,PREDICTING IF THE LEAVE THE COMPANY##


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv(r"C:\Users\Dell\Desktop\dmasm\HR_comma_sep.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.describe


# In[9]:


data.info()


# In[11]:


data.isnull()


# In[12]:


data.isnull().sum()


# In[13]:


#Statistical plots
data_attr=data.iloc[:1:14]
#sns.pairplot(energy_attr,diag_kind='kde')
sns.pairplot(data, diag_kind='kde')


# In[14]:


data.describe().transpose()


# In[15]:


data.corr()


# In[26]:


#Select left data from the column
left=data[data.left==1]
left.shape


# In[27]:


#Select the retained data from the column
retained=data[data.left==0]
retained.shape


# In[28]:


#bar chat showing impact of employees salaries on retention
pd.crosstab(data.salary,data.left).plot(kind='bar')


# In[29]:


#Bar chat showing correlation between department and employee retention
pd.crosstab(data.Department,data.left).plot(kind='bar')


# In[30]:


#Select variable which most impact on employee salary
sel_data = data[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
sel_data.head()


# In[31]:


#Create dummy values of salary variable
dummies = pd.get_dummies(sel_data.salary)


# In[32]:


#concatenate selected variables and dummy variables
data_dummies = pd.concat([sel_data,dummies],axis='columns')


# In[33]:


data_dummies.head()


# In[34]:


#drop salary table
data_dummies.drop('salary',axis='columns',inplace=True)
data_dummies.head()


# In[35]:


#independent and dependent variable
X = data_dummies
y = data.left


# In[36]:


#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3, random_state=0)


# In[38]:


#Logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[39]:


#predict the model
model.predict(X_test)


# In[41]:


#model score
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)


# In[70]:


#K-fold Cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),X,y,cv=4).mean()


# In[42]:


#confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[43]:


#Visualizing the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap (cm,annot=True,fmt=".3f", linewidths=.5, square=True,cmap='Blues_r');
plt.ylabel('y_test');
plt.xlabel('y_pred');
all_sample_title='Accuracy score:{0}'. format(model.score)
plt.title(all_sample_title,size=15)


# In[45]:


fpr=1837/(650+1837)
tpr=7478/(535+7478)


# In[46]:


fpr


# In[47]:


tpr


# In[48]:


#classification report
print(classification_report(y_test,y_pred))


# In[53]:


#Crosstab for categorical indicators
contigency_table1=pd.crosstab(data.Department,data.salary)
contigency_table1


# In[54]:


from scipy.stats.contingency import chi2_contingency
chi_2,p_val,dof,exp_val=chi2_contingency(contigency_table1)


# In[55]:


chi_2


# In[56]:


p_val


# In[57]:


#degree of freedom
dof


# In[58]:


exp_val


# In[66]:


#Barplots for numerical variables
sns.barplot(x='promotion_last_5years',y='satisfaction_level', data=data)
plt.title('promotions VS satisfaction level')


# In[63]:


sns.barplot(x='time_spend_company',y='average_montly_hours', data=data)
plt.title('Company time VS Average hours')


# In[65]:


sns.barplot(x='number_project',y='Work_accident', data=data)
plt.title('Number of project VS Work accident')


# In[68]:


from sklearn.metrics import roc_auc_score,roc_curve
y_pred=model.predict_proba(X_test)[:,1]
fpr,tpr,threshols=roc_curve(y_test, y_pred)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Roc with auc score:{}'.format(roc_auc_score(y_test,y_pred)))
plt.show()


# In[ ]:




