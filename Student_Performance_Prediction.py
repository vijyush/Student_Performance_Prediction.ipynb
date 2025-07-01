#!/usr/bin/env python
# coding: utf-8
TASK-2:
Create a pipeline which predicts the h-index(or score) of the student based on
the features provided to you in the dataset. The train and test data have been
provided in the dataset folder. Please be sure to make an inclusive pipeline which
consists the following:
● Model trained on the training dataset and the predictions for the test
dataset in a .csv file.
● Create a simple interface to use the model using streamlit/gradio.

# In[263]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# ## Analysis of Training Data
# 

# In[264]:


data1 = pd.read_csv('training_data_student.csv')
data1


# In[265]:


data1.shape


# In[266]:


data1.head(21)


# In[267]:


data1.tail(20)


# In[268]:


data1.columns


# In[269]:


data1.info()


# In[270]:


data1.describe()


# In[271]:


data1.describe(include='object')


# In[117]:


data1.isnull().any()


# In[118]:


data1.isnull().sum()


# In[119]:


data1.duplicated().sum()


# In[120]:


pd.pandas.set_option('display.max_rows', null)


# In[121]:


data1


# ### DATA VISUALISATION

# In[122]:


data1.hist(figsize=(15,10),bins=40,alpha=0.5,color='m',edgecolor='black')
plt.show()


# In[123]:


sns.pairplot(data1,hue='Performance Index')
plt.show()


# In[243]:


data1['Performance Index'].hist(alpha=0.9,bins=70,color='yellow',edgecolor='r')
plt.title('student perfomance',color='g',fontsize=15)
plt.xlabel('perfomance Index',color='r',fontsize=12)
plt.ylabel('No.of students',color='r',fontsize=12)
plt.grid('bool'=='False')
plt.figure(figsize=(10,8))

plt.show()


# In[125]:


sns.distplot(data1['Performance Index'],hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')


# In[126]:


numerical_data1=data1.select_dtypes(include='int')
numerical_data1


# In[127]:


numerical_data1=data1.select_dtypes(include='int')
numerical_data1

for i in numerical_data1:
    print(i)


# In[128]:


numerical_data1=data1.select_dtypes(include='int')
numerical_data1

for i in numerical_data1:
    print(i)
    print(data1[i].value_counts())
    print()
    print('********************************************************************')
    


# In[129]:


categorical_data1=data1.select_dtypes(include='object')
categorical_data1


# In[130]:


categorical_data1=data1.select_dtypes(include='object')
categorical_data1

for i in categorical_data1:
    print(i)
    print(data1[i].value_counts())
    print()


# In[131]:


categorical_data1=data1.select_dtypes(include='object').columns
categorical_data1

for i in categorical_data1:
    sns.countplot(data1,x=i,color='b',alpha=0.6,edgecolor='black')
    plt.show()


# ### Dealing Otliers

# In[132]:


sns.boxplot(data1,color='m')

plt.title("Boxplot for data",color='black')
plt.xlabel('x-axis',color='g')
plt.ylabel('Y-axis',color='g')
plt.xticks(rotation=90)
plt.grid()
plt.figure(figsize=(40,20))
plt.show()


# In[275]:


sns.boxplot(data1['Previous Scores'],color='yellow',linewidth=2)
plt.title("Score",color='m')
plt.xlabel('x-axis',color='g')
plt.ylabel('score',color='g')

plt.grid()
plt.figure(figsize=(10,8))
plt.show()


# In[133]:


data1['Extracurricular Activities']=data1['Extracurricular Activities'].map({'Yes':1,'No':0})
data1


# In[134]:


data1.numeric=data1.select_dtypes(include='int')


# In[5]:


correlation_matrix=data1.numeric.corr()


# In[6]:


correlation_matrix


# In[7]:


correlation_matrix['Performance Index']


# In[8]:


sns.heatmap(correlation_matrix,annot=true)
plt.show()


# ## Test Data Analysis 

# In[135]:


data2 = pd.read_csv('test_data_student.csv')
data2


# In[10]:


data2.shape


# In[11]:


data2.dtypes


# In[12]:


data2.info()


# In[13]:


data2.describe()


# In[14]:


data2.describe(include='object')


# In[15]:


data2.isnull().any()


# In[16]:


data2.duplicated().sum()


# ### DATA VISUALISATION

# In[17]:


data2.hist(figsize=(15,10),bins=40,alpha=0.5,color='m',edgecolor='black')
plt.show()


# In[260]:


sns.pairplot(data2,hue='Performance Index')
plt.show()


# In[242]:


data2['Hours Studied'].hist(alpha=0.9,bins=70,color='yellow',edgecolor='r')
plt.title('student perfomance',color='g',fontsize=15)
plt.xlabel('Hours Studied',color='r',fontsize=12)
plt.ylabel('No.of students',color='r',fontsize=12)
plt.grid('bool'=='False')
plt.figure(figsize=(10,8))

plt.show()


# In[241]:


data2['Previous Scores'].hist(alpha=0.9,bins=70,color='yellow',edgecolor='r')
plt.title('student perfomance',color='g',fontsize=15)
plt.xlabel('Marks scored',color='r',fontsize=12)
plt.ylabel('No.of students',color='r',fontsize=12)
plt.grid('bool'=='False')
plt.figure(figsize=(10,8))

plt.show()


# In[20]:


data2[data2['Previous Scores']<75]


# In[21]:


numerical_data2 = data2.select_dtypes(include='int')
numerical_data2


# In[22]:


numerical_data2 = data2.select_dtypes(include='int')
numerical_data2

for i in numerical_data2:
    print(i)


# In[23]:


numerical_data2 = data2.select_dtypes(include='int')
numerical_data2

for i in numerical_data2:
    print(i,'\n')
    print(data2[i].value_counts())
    print()
    print('********************************************************************')



# In[24]:


categorical_data2 = data2.select_dtypes(include='object')
categorical_data2

for i in categorical_data2:
    print(i)


# In[25]:


categorical_data2=data2.select_dtypes(include='object').columns

for i in categorical_data2:
    print(i,'\n')
    print(data1[i].value_counts())
    print()


# In[26]:


categorical_data2=data2.select_dtypes(include='object').columns

for i in categorical_data2:
    sns.countplot(data=data1,x=i,color='m',alpha=0.6,edgecolor='black')
    plt.show()


# ### Dealing the Outliers

# In[27]:


sns.boxplot(data2,color='r')

plt.title("Boxplot for data",color='m')
plt.xlabel('x-axis',color='g')
plt.ylabel('Y-axis',color='g')
plt.xticks(rotation=90)
plt.grid()
plt.figure(figsize=(40,20))
plt.show()


# In[277]:


sns.boxplot(data2['Previous Scores'],color='pink',linewidth=2)
plt.title("Score",color='m')
plt.xlabel('x-axis',color='g')
plt.ylabel('score',color='g')

plt.grid()
plt.figure(figsize=(10,8))
plt.show()


# In[136]:


data2['Extracurricular Activities']=data2['Extracurricular Activities'].map({'Yes':1,'No':0})
data2


# In[137]:


data2_numeric = data2.select_dtypes(include='int')


# In[138]:


correlation_matrix = data2_numeric.corr()


# In[139]:


correlation_matrix


# In[140]:


correlation_matrix['Previous Scores']


# In[141]:


sns.heatmap(correlation_matrix,annot=true)


# ## MODEL BUILDING

# In[142]:


data1


# In[143]:


data2


# In[160]:


data2['Performance Index']=0


# In[161]:


data2


# In[144]:


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[168]:


X_train = data1.drop(columns=['Previous Scores'])
X_train


# In[169]:


y_train=data1['Previous Scores']
y_train


# In[198]:


X_test=data2.drop(columns=['Previous Scores','ID'])
X_test


# In[199]:


y_test=data2['Previous Scores']
y_test


# ### Predictions from Model

# In[200]:


from sklearn.linear_model import LinearRegression


# In[201]:


lm = LinearRegression()


# In[202]:


model=lm
model.fit(X_train,y_train)


# In[203]:


y_pred_train=model.predict(X_train)


# In[204]:


y_pred_train


# In[205]:


plt.scatter(y_train,y_pred_train, edgecolor='black')
plt.title('Training Prediction',color='r',fontsize=18)
plt.xlabel('Actual value',color='g',fontsize=14)
plt.ylabel('Model predicted value',color='g',fontsize=14)
plt.figure(figsize=(20,50))
plt.show()


# In[207]:


y_pred_test=model.predict(X_test)


# In[208]:


y_pred_test


# In[209]:


predictions = lm.predict(X_test)


# In[249]:


plt.scatter(y_test, predictions, edgecolor='black')
plt.title('Test Prediction',color='g',fontsize=18)
plt.xlabel('Actual value',color='m',fontsize=14)
plt.ylabel('Model predicted value',color='m',fontsize=14)
plt.figure(figsize=(20,50))
plt.show()


# ## Measuring the Overfitting of the model
# 

# In[212]:


from sklearn.model_selection import cross_val_score


# In[213]:


cross_val_value_training=cross_val_score(model,X_train,y_train,cv=10)


# In[214]:


cross_val_value_training.mean()


# In[215]:


cross_val_value_testing=cross_val_score(model,X_test,y_test,cv=10)


# In[216]:


cross_val_value_testing.mean()


# ### Model Evaluation

# In[217]:


from sklearn import metrics


# In[218]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R2_Score:',r2_score(y_test,predictions))


# In[219]:


print('MAE:', metrics.mean_absolute_error(y_train,y_pred_train))
print('MSE:', metrics.mean_squared_error(y_train,y_pred_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train,y_pred_train)))
print('R2_Score:',r2_score(y_train,y_pred_train))


# ### Interface Development

# In[220]:


get_ipython().system('pip install streamlit')


# In[234]:


import streamlit as st


# In[248]:


st.title("Student Performance Prediction")
st.write("Enter student features to predict h-index:")

feature_names = X_train.columns.tolist()
user_input = {feature: st.number_input(feature, value=0.0) for feature in feature_names}

if st.button("Predict"):
    features = pd.DataFrame(user_input, index=[0])
    prediction = model.predict(features)
    st.write("Predicted h-index:",prediction[0])


# In[ ]:




