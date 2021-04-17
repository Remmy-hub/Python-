# Python-
Python Projects
#%%
import pandas as pd 
import numpy as np
pd.set_option('display.max_columns',None)
#%%
#Reading the  data 
data= pd.read_csv(r'C:\Users\Lenovo\Documents\Data science\python\HACKATHON\train_CloudCondition.csv',header=0,index_col=0)
data.head()
#%%
#Checking the structure
print(data.shape)
print(data.dtypes)
#%%
#%%
# ambient temp results the same  in farenheit and a combimnation of temp and humidity, However   humidity is important variablehence  not removing that.
data=data.drop(['Temperature (C)'],axis=1)
#%%
data.shape
#%%
data.columns
#%%
# shifting the target variable to the end  
df=data[['Rain_OR_SNOW', 'Apparent Temperature (C)','Humidity',
        'Wind Speed (km/h)', 'Wind Bearing (degrees)',
       'Visibility (km)', 'Pressure (millibars)', 'Condensation',
       'Solar irradiance intensity','Cloud_Condition',]]
df.head()
#%%
# missing values
df.isnull().sum()
#%%
#Treating the  missing values in numerical variables with the mean value
colname1=['Apparent Temperature (C)','Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)']
#%%
for x in colname1:
    df[x].fillna(df[x].mean(),inplace=True)
#%%

df.isnull().sum()
#%%
 #Treating the missing values  in categorical variables with the  mode value.
df['Rain_OR_SNOW'].fillna(df['Rain_OR_SNOW'].mode()[0],inplace=True)
#%%
df.isnull().sum()
#%%
#Converting categorical data  into numerical data .
colname=['Rain_OR_SNOW','Condensation','Cloud_Condition']
#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for x in colname:
    df[x]=le.fit_transform(df[x])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Feature', x)
print('mapping', le_name_mapping)
    

#%%
#Dependent variable -classification
#'Breezy': 0,'Breezy and Dry': 1, 'Breezy and Foggy': 2, 'Breezy and Mostly Cloudy': 3, 'Breezy and Overcast': 4, 'Breezy and Partly Cloudy': 5, 'Clear': 6, 'Dangerously Windy and Partly Cloudy': 7, 'Drizzle': 8, 'Dry': 9, 'Dry and Mostly Cloudy': 10, 'Dry and Partly Cloudy': 11, 'Foggy': 12, 'Humid and Mostly Cloudy': 13, 'Humid and Overcast': 14, 'Humid and Partly Cloudy': 15, 'Light Rain': 16, 'Mostly Cloudy': 17, 'Overcast': 18, 'Partly Cloudy': 19, 'Windy': 20, 
#'Windy and Dry': 21, 'Windy and Foggy': 22, 'Windy and Mostly Cloudy': 23, 'Windy and Overcast': 24, 'Windy and Partly Cloudy': 25
#%%
print(df.head())
print(df.shape)
#%%
df.Cloud_Condition.value_counts()
#%%
#Resampling
df.majority=df[(df.Cloud_Condition==17)|(df.Cloud_Condition==19)|(df.Cloud_Condition==18)]
df.minority=df[(df.Cloud_Condition!=17)|(df.Cloud_Condition!=19)|(df.Cloud_Condition!=18)]
#%%
df.majority.shape
#%%
df.minority.shape
#%%
#Upsampling
from sklearn.utils import resample
df_minority_upsampled =resample(df.minority, replace=True,     # sample with replacement
                                 n_samples=53242,
                                 random_state=10)
#%%

df_upsampled = pd.concat([df.majority, df_minority_upsampled])
#%%
df_upsampled.shape

#%%
# X and Y variables
X=df_upsampled.values[:,0:-1]
Y=df_upsampled.values[:,-1]
#%%
print(X.shape)
print(Y.shape)
#%%
#Standarizing the X variables 
from sklearn.preprocessing import StandardScaler 
Scaler=StandardScaler()
Scaler.fit(X)
X=Scaler.transform(X)
print(X)
#%%
from sklearn.model_selection import train_test_split
# split  the data to train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=10)
#%%
X_train.shape
#%%
Y_test.shape
#Using Decision Tree 
from sklearn.tree import DecisionTreeClassifier
model_DT=DecisionTreeClassifier(criterion='gini',random_state=10,splitter='best')
model_DT.fit(X_train,Y_train)
Y_pred=model_DT.predict(X_test)
#print(Y_pred)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred) # Y_test= Y_actuals and Y_pred = the  predicted Y
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%
print(list(zip(df.columns,model_DT.feature_importances_)))
#%%

from sklearn import tree
with open("model_DecisionTree.txt", "w") as f:

    f = tree.export_graphviz(model_DT, feature_names=df.columns[:-1],out_file=f)
#%%
    #Tuning the DT
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier(criterion="gini",random_state=10,splitter="best",min_samples_leaf=3,max_depth=10)
model_DecisionTree.fit(X_train,Y_train)
Y_pred=model_DecisionTree.predict(X_test)
#print(Y_pred)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%
#Using KNN 
#import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
#model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),metric='euclidean')
#model_KNN.fit(X_train,Y_train)

#Y_pred=model_KNN.predict(X_test)

#%%
from sklearn.metrics import accuracy_score
my_dict={}
for K in range(200,270):# 1:to the sqrt value of no. observation . K= 1 to 30, 31 is exclusive
    model_KNN = KNeighborsClassifier(n_neighbors=K,metric="euclidean")
    model_KNN.fit(X_train, Y_train)
    Y_pred = model_KNN.predict(X_test)
    print ("Accuracy is ", accuracy_score(Y_test,Y_pred), "for K-Value:",K)
    my_dict[K]=accuracy_score(Y_test,Y_pred)
    #%%# TO GET THE  MAXIMUM  VALUE  OF  K
for k in my_dict:
    if my_dict[k]==max(my_dict.values()):
        print(k,":",my_dict[k])
        #%%
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=218,metric='euclidean')
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#%%
# evaluatiing the  model
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)       
 
#%%
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=100, random_state=10,) # n_estimator - denotes  how  many  decision trees .  we can  choose all the  hyper parameter  as in decision tree  to denote the depth and  minimum no. of samples  in each  leaf.  

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc) 
#%%
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=100, random_state=10,min_samples_leaf=3,max_depth=12) # n_estimator - denotes  how  many  decision trees .  we can  choose all the  hyper parameter  as in decision tree  to denote the depth and  minimum no. of samples  in each  leaf.  

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc) 
#%%

#%

#eNSEMBLE MODEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# create the sub models
estimators = []

model1 = DecisionTreeClassifier(criterion='entropy',random_state=10,min_samples_leaf=4,max_depth=8)
estimators.append(('cart', model1))

model2 = KNeighborsClassifier(n_neighbors=260, metric='euclidean')
estimators.append(('knn', model2))
model3= RandomForestClassifier(n_estimators=100, random_state=10,min_samples_leaf=4,max_depth=8)
estimators.append(('Rf',model3))
# create the ensemble model
ensemble = VotingClassifier(estimators) # we have to pass a list  of estiamtors as input  to the voting classifier.  and  the way it shld  be  passed  is  using 2 arguements the  name of the  model  and the  model object. .  
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc) 
#%%
# Predicting  on the client file 
#Reading the  data 
data1= pd.read_csv(r'C:\Users\Lenovo\Documents\Data science\python\HACKATHON\test_CloudCondition.csv',header=0,index_col=0)
data1.head()
#%%
print(data1.shape)
print(data1.dtypes)
#%%
data1=data1.drop(['Temperature (C)'],axis=1)
#%%
data1.shape
#%%
data1.columns
#%%
# missing values
data1.isnull().sum()
#%%
#Converting categorical data  into numerical data .
colname2=['Rain_OR_SNOW','Condensation']
#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for x in colname2:
    data1[x]=le.fit_transform(data1[x])
    #%%
print(data1.head())
#%%
X_test=data1.values[:]
#%%
X=df_upsampled.values[:,0:-1]
#%%
#Standarizing the X variables 
from sklearn.preprocessing import StandardScaler 
Scaler=StandardScaler()
Scaler.fit(X)
X_test=Scaler.transform(X_test)
print(X_test)
#%%
#eNSEMBLE MODEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# create the sub models
estimators = []

model1 = DecisionTreeClassifier(criterion='entropy',random_state=10,min_samples_leaf=4,max_depth=8)
estimators.append(('cart', model1))

model2 = KNeighborsClassifier(n_neighbors=260, metric='euclidean')
estimators.append(('knn', model2))
model3= RandomForestClassifier(n_estimators=100, random_state=10,min_samples_leaf=4,max_depth=8)
estimators.append(('Rf',model3))
# create the ensemble model
ensemble = VotingClassifier(estimators) # we have to pass a list  of estiamtors as input  to the voting classifier.  and  the way it shld  be  passed  is  using 2 arguements the  name of the  model  and the  model object. .  
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)

#%%
Y_pred.shape
#%%
data1['Cloud_Condition']=Y_pred
#%%
data1.head()
#%%
#'Breezy': 0,'Breezy and Dry': 1, 'Breezy and Foggy': 2, 'Breezy and Mostly Cloudy': 3, 'Breezy and Overcast': 4, 'Breezy and Partly Cloudy': 5, 'Clear': 6, 'Dangerously Windy and Partly Cloudy': 7, 'Drizzle': 8, 'Dry': 9, 'Dry and Mostly Cloudy': 10, 'Dry and Partly Cloudy': 11, 'Foggy': 12, 'Humid and Mostly Cloudy': 13, 'Humid and Overcast': 14, 'Humid and Partly Cloudy': 15, 'Light Rain': 16, 'Mostly Cloudy': 17, 'Overcast': 18, 'Partly Cloudy': 19, 'Windy': 20, 
#'Windy and Dry': 21, 'Windy and Foggy': 22, 'Windy and Mostly Cloudy': 23, 'Windy and Overcast': 24, 'Windy and Partly Cloudy': 25
#%%
data1['Cloud_Condition']=data1['Cloud_Condition'].map({0:'Breezy',1:'Breezy and Dry',2:'Breezy and Foggy',3:'Breezy and Mostly Cloudy',4:'Breezy and Overcast',5:'Breezy and Partly Cloudy',6:'Clear',7:'Dangerously Windy and Partly Cloudy',
                                                      8:'Drizzle',9:'Dry',10:'Dry and Mostly Cloudy',11:'Dry and Partly Cloudy',12:'Foggy',13:'Humid and Mostly Cloudy',14:'Humid and Overcast',15:'Humid and Partly Cloudy',16:'Light Rain',17:'Mostly Cloudy',18:'Overcast',
                                                      19:'Partly Cloudy',20:'Windy',21:'Windy and Dry',22:'Windy and Foggy',23:'Windy and Mostly Cloudy',24:'Windy and Overcast',25:'Windy and Partly Cloudy'})
#%%
data1.head()
#%%
data1.columns
#%%
data1=data1.drop(['Rain_OR_SNOW', 'Apparent Temperature (C)', 'Humidity',
       'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
       'Pressure (millibars)', 'Condensation', 'Solar irradiance intensity'],axis=1)
#%%
data1.head()
#%%
data1.to_csv(r'C:\Users\Lenovo\Documents\Data science\python\HACKATHON\Cloud_condition_output13.csv',header=True,index=True)
