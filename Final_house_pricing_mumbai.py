#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Mumbai House Prices.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df["Amount"] = df['price'].astype(str) +" "+ df["price_unit"]
df.head()


# In[6]:


def fun(x):
    if 'Cr' in x or 'cr' in x:
        s=str(x).split(" ")[0]
        s1=str(int(float(s)*100))
        return s1
    else:
        s=str(x).split(" ")[0]
        return s


# In[7]:


df['Amount']=df['Amount'].apply(fun)


# In[8]:


df.head()


# In[9]:


df.drop(columns=['price','price_unit'],inplace=True)


# In[10]:


df.head()


# In[11]:


df['type'].value_counts()


# In[12]:


row_indices, _ = np.where(df == 'Penthouse')

# Drop rows where 'Built Area' is present
df = df.drop(index=row_indices)


# In[13]:


row_indices, column_indices = np.where(df== 'Penthouse')
print("Row indices with 'Built Area':", row_indices)
print("Column indices with 'Built Area':", column_indices)


# In[14]:


types = {'Apartment':0,'Studio Apartment':1,'Villa':2,'Independent House':3}
df['type']=df['type'].replace(types)


# In[15]:


df.head()


# In[16]:


df['region'].value_counts()


# In[17]:


df['region'].value_counts()[df['region'].value_counts()>20]


# In[18]:


df['status'].value_counts()


# In[19]:


df['status']=df['status'].replace(['Ready to move','Under Construction'],[1,0])


# In[20]:


df.head()


# In[21]:


df['age'].value_counts()


# In[22]:


df.drop(columns=['age'],inplace=True)


# In[23]:


df.head()


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[25]:


X = df[['bhk','type','area','status']]
y = df['Amount']


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=110)


# In[27]:


lm = LinearRegression()


# In[28]:


lm.fit(X_train,y_train)


# In[29]:


c = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
c


# In[30]:


p=lm.predict(X_test)


# In[31]:


p


# In[32]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, p))
print('MSE:', metrics.mean_squared_error(y_test, p))
print('RMaSE:', np.sqrt(metrics.mean_squared_error(y_test, p)))


# In[33]:


pd.DataFrame(y_test)


# In[34]:


df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')


# In[35]:


df['Amount'].dtype


# In[36]:


actual_values = y_test
predicted_values= p
results = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})
results = results.round(0)
results


# In[37]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, p)
print(f"R-squared: {r2}")


# In[41]:


columns_used = ['bhk','type','area','status']
user_input = {}
for column in columns_used:
    user_input[column] = input(f"Enter value for {column}: ")
user_input_df = pd.DataFrame([user_input], columns=columns_used)
user_prediction = lm.predict(round(user_input_df))

# Display the predicted rent
print(f"The predicted rent is: {user_prediction[0]}")


# In[47]:


a=df.copy()


# In[48]:


df.head()


# In[55]:


top_regions = df['region'].value_counts().head(5).index.tolist()

# Filter the DataFrame to include only rows with the top 5 regions
filtered_df = df[df['region'].isin(top_regions)]


# In[56]:


filtered_df


# In[57]:


filtered_df['region'].value_counts()


# In[78]:


regions={'Thane West':0,'Mira Road East':1,'Dombivali':2,'Kandivali East':3,'Kharghar':4}
filtered_df['region']=filtered_df['region'].replace(regions)


# In[79]:


filtered_df


# In[105]:


X = filtered_df[['bhk','type','area','region','status']]
y = filtered_df['Amount']


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=110)


# In[107]:


lr = LinearRegression()


# In[108]:


lr.fit(X_train,y_train)


# In[109]:


c = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
c


# In[110]:


p1=lr.predict(X_test)
p1


# In[111]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, p1))
print('MSE:', metrics.mean_squared_error(y_test, p1))
print('RMaSE:', np.sqrt(metrics.mean_squared_error(y_test, p1)))


# In[112]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, p)
print(f"R-squared: {r2}")


# In[113]:


filtered_df.head()


# In[117]:


columns_used = ['bhk','type','area','region','status']
user_input = {}
for column in columns_used:
    user_input[column] = input(f"Enter value for {column}: ")
user_input_df = pd.DataFrame([user_input], columns=columns_used)
user_prediction = lr.predict(round(user_input_df))

# Display the predicted rent
print(f"The predicted rent is: {user_prediction[0]}")


# In[118]:


actual_values = y_test
predicted_values= p1
results = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})
results = results.round(0)
results


# In[119]:


residuals = y_test - p1
plt.scatter(p, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# In[120]:


plt.scatter(y_test, p1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='r')
plt.show()


# In[121]:


actual_values = y_test
predicted_values = p1  

plt.figure(figsize=(10, 6))
sns.scatterplot(x=actual_values, y=predicted_values)
plt.xlabel("Actual Values (Rent)")
plt.ylabel("Predicted Values (Rent)")
plt.title("Actual vs. Predicted Values")

plt.plot([min(actual_values), max(actual_values)], 
         [min(actual_values), max(actual_values)], color='red')
plt.show()


# In[ ]:




