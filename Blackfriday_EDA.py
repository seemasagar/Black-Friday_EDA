#!/usr/bin/env python
# coding: utf-8

# # Black friday Dataset EDA and Feature Engeering cleaning and preparing the data for model training

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Problem statement
# A retail company "ABC Private Limited" wants to understand the custmer purchase behaviour(specially, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contain customer demographics(age, gender, marital status, city_type, stay_in_current_city), product details(product_id and product category) and total purchase_amount from last month.
# 
#  Now, they want to build a model to predict the purchase amount of  customer against various products which will help them to create personalized offer for customers against different products.

# In[2]:


#importing the dataset
df_train=pd.read_csv(r"C:\Users\seema sagar\Downloads\Blackfriday_train.csv", encoding='latin-1')


# In[3]:


df_train


# In[4]:


## Import the test data 
df_test=pd.read_csv(r"C:\Users\seema sagar\Downloads\Blackfriday_test.csv", encoding='latin-1')


# In[5]:


df_test.head()


# #merge both train and test data
# #point to note whenever train and test data is given always
# merge the two data first and for kaggle compition also
# 

# In[8]:



df= df_train.append(df_test)
df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.drop('User_ID', axis=1, inplace=True)


# In[12]:


df.head()


# In[13]:


#Handling categorical feature gender
df['Gender']=df['Gender'].map({'F':0, 'M':1})


# In[14]:


df.head()


# In[15]:


# Handle categorical feature age
df['Age'].unique()


# In[16]:


#pd.get_dummies(df['Age'], drop_first=True)
df['Age']=df['Age'].map({'0-17':1, '18-25':2, '26-35':3,'36-45':4,'46-50':5,'51-55':6})


# In[17]:


df.head()


# In[18]:


#SECOND TECHNIQUE
from sklearn import preprocessing
#label_encoder object knows how to understand word labels
label_encoder= preprocessing.LabelEncoder()
#Encode labels in column 'Age'
df['Age']=label_encoder.fit_transform(df['Age'])
df['Age'].unique()


# In[19]:


df.head()


# In[20]:


#fixing categorical city _category

df_city= pd.get_dummies(df['City_Category'],drop_first=True)


# In[21]:


df_city.head()


# In[22]:


df=pd.concat([df,df_city],axis=1)


# In[23]:


df.head()


# In[24]:


df.drop('City_Category',axis=1)


# In[25]:


df.head()


# In[26]:


#missing Values
df.isnull().sum()


# In[27]:


#Focus on replacing missing values
#this is discrete feature
df['Product_Category_2'].unique()


# In[28]:


df['Product_Category_2'].value_counts()


# In[29]:


## Replace the missing value with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[30]:


df['Product_Category_2'].isnull().sum()


# In[32]:


##product category 3 replace missing values
df['Product_Category_3'].unique()


# In[34]:


df['Product_Category_3'].value_counts()


# In[35]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[36]:


df.head()


# In[37]:


df['Stay_In_Current_City_Years'].unique()


# In[40]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[41]:


df.head()


# In[42]:


df.info()


# In[43]:


#convert object into integers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df.head()


# In[44]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[45]:


df.info()


# In[50]:


##visualization of age verse purchase
sns.barplot('Age','Purchase',hue='Gender', data=df)


# # Purchasing of men is high then women

# In[51]:


#visualization of purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender', data=df)


# In[52]:


sns.barplot('Product_Category_1','Purchase',hue='Gender', data=df)


# In[54]:


sns.barplot('Product_Category_2','Purchase',hue='Gender', data=df)


# In[55]:


sns.barplot('Product_Category_3','Purchase',hue='Gender', data=df)


# In[56]:


df.head()


# In[57]:


##Feature Scaling
df_test=df[df['Purchase'].isnull()]


# In[68]:


df_train=df[~df['Purchase'].isnull()]


# In[97]:


X=df_train.drop('Purchase', axis=1)


# In[98]:


X.head()


# In[99]:


X.shape


# In[100]:


y=df_train['Purchase']


# In[101]:


y.shape


# In[105]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[106]:


X_train.drop('Product_ID', axis=1, inplace=True)
X_test.drop('Product_ID', axis=1, inplace=True)


# In[107]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:





# In[ ]:




