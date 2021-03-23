#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas_profiling as pp
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.model_selection import train_test_split


# ### Importing Data: 

# In[2]:


raw_data = pd.read_csv("bank-full.csv", sep=";")
raw_data


# ### Exploratory Data Analysis:

# In[3]:


raw_data.info()


# In[4]:


# Categorial data
cat_data = raw_data.select_dtypes(exclude='number')
cat_data


# In[5]:


# Numerical data
num_data = raw_data.select_dtypes(include='number')
num_data


# In[6]:


raw_data.describe().T


# #### the value of pdays if -1 means the person is not contacted. So we can make a new feature i.e. if a person was contacted or not.

# In[7]:


raw_data["pdays_no_contact"] = (raw_data["pdays"] == -1) * 1
raw_data


# In[8]:


# Checking how many values are there in categorical data
for col in cat_data:
    print(col, "\n")
    print(cat_data[col].value_counts())
    print("_____________","\n\n")


# #### Is there a balance between yes and no in our data? 

# In[9]:


plt.rcParams["figure.figsize"] = (5,5)
raw_data["y"].value_counts().plot.bar()


# ###### Observation: We can see that the data is imbalanced as there are more No values than Yes values therefor the model may be biased

# In[10]:


# Visualizing the data to see if there is any relation of the dependant variable with the independant variables
not_plot = ["balance", "duration", "pdays", "y"] # not plotting this due to high data and y as it is dependant

for col in raw_data.columns:
    if col not in not_plot:  
        plt.rcParams["figure.figsize"] = (18,6)
        pd.crosstab(raw_data[col],raw_data.y).plot(kind="bar")


# In[11]:


# Quick EDA. The file with all the EDA can be found in the root folder
EDA_report= pp.ProfileReport(raw_data)
EDA_report.to_file(output_file='EDA_report.html')


# In[12]:


# Changing the Binary categorical data to 0 & 1
data = raw_data.copy()
data["housing"] = data["housing"].map({"yes":1, "no":0})
data["loan"] = data["loan"].map({"yes":1, "no":0})
data["y"] = data["y"].map({"yes":1, "no":0})
data["default"] = data["default"].map({"yes":0, "no":1})
data


# In[13]:


# Checking the correlation
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(data.corr(), annot=True, ax=ax)


# In[14]:


# getting the dummies for the rest of the categorical data

data = pd.get_dummies(data, columns=["job", "marital", "education", "housing", "loan", "contact", "month", "poutcome"], drop_first=True)


# ### Training the logistic regression model:

# In[15]:


x = data.loc[:,data.columns != "y"]
y = data["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[16]:


scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[17]:


log_reg_model = LogisticRegression()
log_reg_model.fit(x_train,y_train)


# #### Prediction:

# In[18]:


x = scaler.fit_transform(x)
y_pred = log_reg_model.predict(x)
data["predicted"] = y_pred
data


# In[51]:


check_prediction = data[["y", "predicted"]]
check_prediction


# #### Confusion marix:

# In[21]:


confusion_matrix = metrics.confusion_matrix(y, y_pred)
confusion_matrix


# In[22]:


pd.crosstab(y, y_pred)


# #### Accuracy:

# In[23]:


train_accuracy = log_reg_model.score(x_train,y_train)
test_accuracy = log_reg_model.score(x_test,y_test)


print(f'''Train Accuracy: {train_accuracy}
Test Accuracy: {test_accuracy}''')


# In[24]:


print(classification_report(y, y_pred))


# In[25]:


Logit_roc_score=roc_auc_score(y,log_reg_model.predict(x))
Logit_roc_score


# In[26]:


fpr, tpr, thresholds = roc_curve(y,log_reg_model.predict_proba(x)[:,1]) 
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()  


# In[27]:


y_prob1 = pd.DataFrame(log_reg_model.predict_proba(x_train)[:,1]) 
y_prob1


# ## Using the logit model:

# In[28]:


logit = sm.Logit(y,x) 


# In[29]:


logit_model = logit.fit()


# In[30]:


logit_model.summary()


# In[31]:


y_pred_logit = logit_model.predict()
y_pred_logit


# In[32]:


logit_model.pred_table()


# In[33]:


cm_df = pd.DataFrame(logit_model.pred_table())
cm_df.columns = ["Predicted 0", "Predicted 1"]
cm_df = cm_df.rename(index={0:"Actual 0", 1:"Actual 1"})
cm_df


# In[34]:


logit_accuracy = (cm_df.iloc[0,0] + cm_df.iloc[1,1])/ data.shape[0]
logit_accuracy


# ### Backward Elimination:

# #### Removing insignificant data:

# In[35]:


significant_features = []

p = logit_model.pvalues.to_dict()

for key, val in p.items():
    if val <= 0.05:
        num = key
        significant_features.append(int(num[1:])-1)
        
significant_features


# In[36]:


x = pd.DataFrame(x)
x = x[significant_features]
x


# #### Model_2:

# In[37]:


logit = sm.Logit(y,x)
logit_model_2 = logit.fit()


# In[38]:


logit_model_2.summary()


# ###### Observation: the model has one insignificant feature

# In[39]:


y_pred_logit_2 = logit_model_2.predict()
y_pred_logit_2


# In[40]:


logit_model_2.pred_table()


# In[41]:


cm_df = pd.DataFrame(logit_model_2.pred_table())
cm_df.columns = ["Predicted 0", "Predicted 1"]
cm_df = cm_df.rename(index={0:"Actual 0", 1:"Actual 1"})
cm_df


# In[42]:


logit_accuracy = (cm_df.iloc[0,0] + cm_df.iloc[1,1])/ data.shape[0]
logit_accuracy


# #### Removing insignificant data:

# In[43]:


new_significant = []

for i in significant_features:
    if i != 11:
        new_significant.append(i)


# In[44]:


x = x[new_significant]
x


# #### model_3:

# In[45]:


logit = sm.Logit(y,x)
logit_model_3 = logit.fit()


# In[46]:


logit_model_3.summary()


# In[47]:


logit_model_3.pred_table()


# In[48]:


cm_df = pd.DataFrame(logit_model_3.pred_table())
cm_df.columns = ["Predicted 0", "Predicted 1"]
cm_df = cm_df.rename(index={0:"Actual 0", 1:"Actual 1"})
cm_df


# In[49]:


logit3_accuracy = (cm_df.iloc[0,0] + cm_df.iloc[1,1])/ data.shape[0]
logit3_accuracy


# In[50]:


all_cols = list(data.columns)

final_cols = []

for i in new_significant:
    final_cols.append(all_cols[i])

print(f"The significant features for our logit model are {final_cols}")


# ### Conclusion:

# ###### The accuracy of the models are given in the below table`

# In[498]:


model_table = pd.DataFrame({
    "Model": ["logistic regression model", "logit model"],
    "accuracy": [test_accuracy, logit3_accuracy]
})
model_table


# In[ ]:




