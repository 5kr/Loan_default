#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#Sklearn package's data splitting function which is based on random function
from sklearn.model_selection import train_test_split

#For logistic Regression model
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# calculate accuracy measures and confusion matrix
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# # Data Dictionary

# | Variables        | Type           | Description  |
# | ------------- |:-------------:| -----:|
# | Gender      | Categorical | Male or Female |
# | LoanOffered      | Indicator      |   Whether the Bank approached the customer for the loan: 1 indicates Bank did |
# | Job | Categorical      |    Unskilled, Skilled, Management |
# | CreditScore | Numeric | Credit score during loan approval; high score indicates better borrower|
# | EMIRatio | Numeric | Estimated EMI/Total assets during granting of loan|
# | Status | Categorical | Default or Not|
# | Purpose | Categorical | Purpose for borrowing: Car, Consumer durables or Personal|
# | Dependents | Integer | Number of dependents|

# In[2]:


df = pd.read_csv('default2.csv')
df.head()


# In[3]:


df.columns


# In[4]:


#df.drop('CreditScore',axis=1, inplace=True)


# In[5]:


df['LoanOffered'] = df['LoanOffered'].astype('category')


# In[6]:


#Tag default - positive class based on class on interest
df['Status'].value_counts()


# In[7]:


#Class of Interest is customers who default
df['Target']= np.where(df['Status']=='Default',1,0)
df['Target'].value_counts()


# In[8]:


df.drop(['Status'],axis=1, inplace=True)


# ## Exploratory Data Analysis

# In[9]:


df.columns


# In[10]:


sns.boxplot(x=df['Target'], y=df['EMIRatio'])


# In[11]:


ct = pd.crosstab(index=df['Gender'],
           columns=df['Target'],
           values = df['Target'],
           aggfunc='count',
           normalize='index').round(3)


# In[12]:


ct.plot(kind='bar',stacked=True)


# ## Model Building

# In[13]:


X = df.drop(['Target'], axis=1)
Y = df[['Target']]


# In[14]:


X = pd.get_dummies(X, drop_first=True)


# In[15]:


import statsmodels.formula.api as SM


# In[16]:


def vif_cal(input_data):
    x_vars=input_data
    xvar_names=input_data.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=SM.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)


# In[17]:


vif_cal(X)








# In[18]:


#Logistic regression using stats model - method 'bfgs' - Broyden–Fletcher–Goldfarb–Shanno optimization
import statsmodels.api as sm

import random
random.seed(10)
logreg = sm.Logit(Y, sm.add_constant(X) )
lg = logreg.fit()


# In[19]:


#Summary of logistic regression
print(lg.summary())


# In[20]:


## Collect the coef in a dataframe
lgcoef = pd.DataFrame(lg.params, columns=['coef'])

## Calculate Odds Ratio
lgcoef['Odds_ratio']=np.exp(lgcoef.coef)
lgcoef


# In[21]:


#Calculate Probability
lgcoef['probability'] = lgcoef['Odds_ratio']/(1+lgcoef['Odds_ratio'])
lgcoef


# In[22]:


#Calculate p value
lgcoef['pval']=lg.pvalues

#Display rounded to 2 decimal points
pd.options.display.float_format = '{:.2f}'.format
lgcoef


# In[23]:


#Sort by descending order of odds ratio 
lgcoef = lgcoef.sort_values(by="Odds_ratio", ascending=False)

#Filter to display only variables with significant p value
pval_filter = lgcoef['pval']<=0.1
lgcoef[pval_filter]


# # Predict based on the model 

# In[24]:


X1 = sm.add_constant(X)
ypred_prob = lg.predict(X1)
ypred  = (ypred_prob > 0.5).astype(int)
ypred


# In[28]:


# Append Y_pred_prob to original df and write a csv file for analysis
df['pred_prob'] = ypred_prob
df.head()


# In[29]:


df_pred = df[['Target','pred_prob']]
df_pred.head()


# In[31]:


#Boxplot of predicted probabilities
sns.boxplot(x=df['Target'],y=df['pred_prob'])


# In[32]:


#Import to csv file for further analysis in excel
df.to_csv('Predicted_data_default.csv')


# ### change cutoff values in excel for demo

# In[33]:


#Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


# In[34]:


confusion_matrix(Y,ypred)


# In[37]:


accuracy_score(Y,ypred)


# In[39]:


recall_score(Y,ypred)


# In[40]:


precision_score(Y,ypred)


# In[41]:


print(classification_report(Y,ypred))


# In[42]:


#AUC Value
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y,ypred)
logit_roc_auc


# In[44]:


#Plotting the ROC Curve
fpr, tpr, threshold = roc_curve(Y,ypred)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])


# In[58]:


# Predict the probability value for X
pred_proba_df = pd.DataFrame(ypred_prob)


# In[57]:


# Use Cut-off value to predict the recall_score and accuracy_score.
cutoff_list = [0.2,0.8]
for i in cutoff_list:
    print ('\n******** For i = {} ******'.format(i))
    y_pred = pd.DataFrame(np.where(pred_proba_df > i, 1, 0))
    test_recall_score = metrics.recall_score(Y, y_pred)
    test_acu_score = metrics.roc_auc_score(Y, y_pred)
    test_precision_score = metrics.precision_score(Y, y_pred)
    print('Our testing recall is {:.2f}'.format(test_recall_score))
    print('Our Accuracy score is {:.2f}'.format(test_acu_score))
    print('Our Precision score is {:.2f}'.format(test_precision_score))






