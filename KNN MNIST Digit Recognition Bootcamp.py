#!/usr/bin/env python
# coding: utf-8

# ## basics of the KNN MNIST project

# In[29]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[30]:


#extracting datafiles which is already downloaded & uploaded in jupyter notebook home page
dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[31]:


x=dfx.values
y=dfy.values
#now setting the data in given dimesions 
X=x[:,1:]
Y=y[:,1:].reshape((-1,))    #by reshape we reshape/remove the column "given dimension or column number"

print(X)
print(X.shape)
print(y.shape)


# In[32]:


plt.scatter(X[:,0],X[:,1],c=Y)   #this will create a scatterplot 


# In[33]:


#this will mark a particular point in the given scatterplot 
query_X=np.array([2,3])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(query_X[1],query_X[0],color='brown')
plt.show()          


# In[49]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))             #distance function between the two points 

def knn(X,Y,queryPoint,k=7):                          #this is knn function
    
    vals=[]                              #an empty array
    
    m=X.shape[0]
    
    for i in range(m):                          #itrating "i" over the entire number of columns
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)                     #here we are sorting vals (the nearest values)
    vals=vals[:k]
        
    vals=np.array(vals)
        
    #print(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)    
    print(new_vals)
    
    index=new_vals[1].argmax()        #using argmax to find the maximum 
    pred=new_vals[0][index]
    
    return pred


# In[50]:


knn(X,Y,query_X)        #now checking the K's nearest neighbours or KNN's prediction


# ## MNIST DATASET

# In[51]:


#extracting datafiles which is already downloaded & uploaded in jupyter notebook home page
df=pd.read_csv('train.csv')
print(df.shape)
print(df.columns)
df.head(15)


# In[52]:


#creating numpy array here
data=df.values
print(data.shape)
print(type(data))


# In[53]:


X=data[:,1:]
Y=data[:,0]

print(X.shape,Y.shape)          #number of rows and columns in given dimensions


# In[54]:


split=int(0.8*X.shape[0])           #spliting the integers here
print(split)


# In[55]:


#defining all the training and testing data according to the above table 
X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[56]:


#now defing the draw image function
def drawImg(sample):
    Img=sample.reshape((28,28))
    plt.imshow(Img)
    plt.show()


# ### checking the above functions with few examples  

# In[57]:


drawImg(X_train[6])
print(Y_train[6])


# In[58]:


drawImg(X_train[11])
print(Y_train[11])


# In[59]:


drawImg(X_train[10])
print(Y_train[10])


# In[60]:


drawImg(X_train[8])
print(Y_train[8])

