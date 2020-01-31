#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


data = np.loadtxt("notas_andes.dat", skiprows=1)


# In[143]:


X = data[:,:4]
Y = data[:,4]

X_random = np.zeros([69,4])
Y_random = np.zeros(69)


regresion = sklearn.linear_model.LinearRegression()

B = np.zeros([1000,4])

def betas(X,Y):
    for i in range(len(X)):
        indices = np.random.choice(np.arange(69))    
        X_random[i,:] = X[indices,:]    
        Y_random[i] = Y[indices]
    
    regresion.fit(X_random, Y_random)
    return regresion.coef_

for i in range(1000):
    B[i,:] = betas(X,Y)

    
plt.figure(figsize =(5,5))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(B[:,i], bins=20)
    plt.title(r"$\beta_{}={:.2f}\pm{:.2f}$".format(i+1,np.mean(B[:,i]),np.std(B[:,i])))

plt.tight_layout()


plt.savefig("bootstrapping.png", bbox_inches='tight')



  


# In[23]:





# In[ ]:




