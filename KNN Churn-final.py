#!/usr/bin/env python
# coding: utf-8

# In[85]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


df=pd.read_csv('churn.csv')


# In[87]:


df


# In[88]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# __detecting numbers in string containing column, and changing them to NaN, good for cleaning__

# In[89]:


# Detecting numbers 
#cnt=0
#for row in df['gender']:
 #   try:
  #      int(row)
    #    df.loc[cnt, 'gender']=np.nan
    #except ValueError:
     #   pass
    #cnt+=1


# __check where Nan values are in three different forms__

# In[90]:


#list(map(tuple, np.where(np.isnan(x))))


# In[91]:


df.isnull().stack()[lambda x: x].index.tolist()


# In[92]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[93]:


df.Churn.value_counts()


# In[94]:


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


# In[95]:


df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)


# In[96]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[97]:


df = df[df["TotalCharges"].notnull()]
df = df.reset_index()[df.columns]


# In[98]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[99]:


#replace 'No internet service' to No for the following columns
replace_cols = [ 'OnlineSecurity', 'MultipleLines', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    df[i]  = df[i].replace({'No internet service' : 'No'})
    df[i]  = df[i].replace({'No phone service' : 'No'})


#Tenure to categorical column
def tenure_lab(df) :
    
    if df["tenure"] <= 12 :
        return "1"
    elif (df["tenure"] > 12) & (df["tenure"] <= 24 ):
        return "2"
    elif (df["tenure"] > 24) & (df["tenure"] <= 48) :
        return "3"
    elif (df["tenure"] > 48) & (df["tenure"] <= 60) :
        return "4"
    elif df["tenure"] > 60 :
        return "5"
df["tenure_group"] = df.apply(lambda df:tenure_lab(df),
                                      axis = 1)


# In[100]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[102]:


churndf=df[df.Churn=='Yes']
nochurndf=df[df.Churn=='No']


# In[103]:


churndf


# In[104]:


from matplotlib.gridspec import GridSpec
def plotpiesbychurn(column):
        plt.figure(1, figsize=(20,10))
        the_grid = GridSpec(2, 2)
        valch = churndf[column].value_counts().values.tolist()
        labelsch  = churndf[column].value_counts().keys().tolist()
        plt.subplot(the_grid[0, 0], aspect=1)
        plt.pie(valch, labels=labelsch, autopct='%1.1f%%',
        shadow=True, startangle=90)
        plt.title("Churned")
        plt.suptitle(column + " distribution in customer churn ",y=1)
        valnch = nochurndf[column].value_counts().values.tolist()
        labelsnch  = nochurndf[column].value_counts().keys().tolist()
        plt.subplot(the_grid[0, 1], aspect=1)
        plt.pie(valnch, labels=labelsnch, autopct='%1.1f%%',
        shadow=True, startangle=90)
        plt.title("Didn't Churn")


        return plt.show()



# In[105]:


for column in df.columns:
    if ((column !='customerID') & (column !='Churn') & (column !='MonthlyCharges') & (column !='TotalCharges')):   
        plotpiesbychurn(column)


# In[107]:


#changes genders to binary with male=1 and famale=0
cnt=0
for row in df['gender']:
    if (df.loc[cnt, 'gender']=="Male"):
         df.loc[cnt, 'gender']=1
    else:
        df.loc[cnt, 'gender']=0
    cnt+=1
#changes SeniorCitizen to binary with yes=1 and no=0

#changes Partner to binary with yes=1 and no=0
cnt=0
for row in df['Partner']:
    if (df.loc[cnt, 'Partner']=="Yes"):
         df.loc[cnt, 'Partner']=1
    else:
        df.loc[cnt, 'Partner']=0
    cnt+=1

#changes PhoneService to binary with yes=1 and no=0
cnt=0
for row in df['PhoneService']:
    if (df.loc[cnt, 'PhoneService']=="Yes"):
         df.loc[cnt, 'PhoneService']=1
    else:
        df.loc[cnt, 'PhoneService']=0
    cnt+=1
#changes Churn to binary with yes=1 and no=0
cnt=0
for row in df['Churn']:
    if (df.loc[cnt, 'Churn']=="Yes"):
         df.loc[cnt, 'Churn']=1
    else:
        df.loc[cnt, 'Churn']=0
    cnt+=1


# In[108]:


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[110]:


df["gender"] = df["gender"].astype(float)
df["Partner"] = df["Partner"].astype(float)
df["tenure_group"] = df["tenure_group"].astype(float)
df["PhoneService"] = df["PhoneService"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)


np.where(pd.isnull(df)) #returns the row and column indices where the value is NaN:


# In[111]:


x=df[['gender','SeniorCitizen','Partner','tenure_group','PhoneService','MonthlyCharges','TotalCharges']].values


# In[112]:


xdf=df[['gender','SeniorCitizen','Partner','tenure_group','PhoneService','MonthlyCharges','TotalCharges','Churn']]


# In[113]:


xdf


# In[114]:


np.where(pd.isnull(xdf)) #returns the row and column indices where the value is NaN:


# In[115]:


np.set_printoptions(suppress=True)
x


# In[116]:


y = df['Churn'].values


# In[117]:


y


# In[118]:


x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))


# __when using KNN, all values should be numerical (continous or ordinal) bcoz it has to be numpy array, the process of changing categorical to ordinal is called one hot encoding__
# 
# Hot encoding turns a column with n caegories to n columns. Dummy coding just assigns a number to the parameters.

# In[119]:


x


# In[120]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[121]:


from sklearn.neighbors import KNeighborsClassifier


# In[122]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[123]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[124]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[125]:


k = 6
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh
yhat = neigh.predict(X_test)
yhat[0:5]
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[126]:


Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[127]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[128]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# __Decision Tree__

# In[129]:


from sklearn.model_selection import train_test_split


# In[130]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)


# In[131]:


from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy",max_depth=2)
drugTree # it shows the default parameters


# In[132]:


drugTree.fit(X_trainset,y_trainset)


# In[133]:


predTree = drugTree.predict(X_testset)


# In[134]:


print (predTree [0:5])


# In[135]:


print (predTree [0:5])
print (y_testset [0:5])


# In[136]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[139]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
columns = xdf.columns[0:7]
dot_data = StringIO()
export_graphviz(drugTree, out_file=dot_data,feature_names = columns, class_names=["Not churn","Churn"], 
                filled=True, rounded=True,
                special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# __Logistic Regression__

# In[140]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[141]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[142]:


yhat = LR.predict(X_test)
yhat


# In[143]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[144]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[145]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[146]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# In[ ]:




