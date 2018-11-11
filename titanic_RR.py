
# coding: utf-8

# # 1.Project understanding
# (Source: wikipedia)
# RMS Titanic sank in the early morning of 15 April 1912 in the North Atlantic Ocean, four days into the ship's maiden voyage from Southampton to New York City. The largest passenger liner in service at the time, Titanic had an estimated 2,224 people on board when she struck an iceberg at around 23:40 (ship's time)[a] on Sunday, 14 April 1912. Her sinking two hours and forty minutes later at 02:20 (ship's time; 05:18 GMT) on Monday, 15 April, resulted in the deaths of more than 1,500 people, which made it one of the deadliest peacetime maritime disasters in history.
# 
# Titanic received six warnings of sea ice on 14 April but was travelling near her maximum speed when her lookouts sighted the iceberg. Unable to turn quickly enough, the ship suffered a glancing blow that buckled her starboard side and opened five of her sixteen compartments to the sea. Titanic had been designed to stay afloat with four of her forward compartments flooded but no more, and the crew soon realised that the ship would sink. They used distress flares and radio (wireless) messages to attract help as the passengers were put into lifeboats. In accordance with existing practice, Titanic's lifeboat system was designed to ferry passengers to nearby rescue vessels, not to hold everyone on board simultaneously; therefore, with the ship sinking rapidly and help still hours away, there was no safe refuge for many of the passengers and crew. Compounding this, poor management of the evacuation meant many boats were launched before they were completely full. 
# 
# As a result, when Titanic sank, over a thousand passengers and crew were still on board. Almost all those who jumped or fell into the water either drowned or died within minutes due to the effects of cold shock and incapacitation. RMS Carpathia arrived on the scene about an hour and a half after the sinking and rescued the last of the survivors by 09:15 on 15 April, some nine and a half hours after the collision. The disaster shocked the world and caused widespread outrage over the lack of lifeboats, lax regulations, and the unequal treatment of the three passenger classes during the evacuation. Subsequent inquiries recommended sweeping changes to maritime regulations, leading to the establishment in 1914 of the International Convention for the Safety of Life at Sea (SOLAS), which still governs maritime safety today. 
# 
# In this project, we are going to predict the survival rate based on previous data

# In[4]:


import pandas as pd


# In[221]:


#Import file
df=pd.read_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\titanic.csv')


# # 2.Data understanding

# In[222]:


df.head()


# In[223]:


df.tail()


# In[224]:


df.columns


# In[225]:


df.shape


# In[226]:


df.info()


# Missing values: 
# 263 or 20% missing values in "age"
# 1014 or 77.4% missing values in "Cabin"
# 823 or 63% missing values in "boat"
# 1188 or 91% missing values in "body"
# 564 or 43% missing values in "home.dest"
# 
# large amount of missing values in "cabin", "boat","body,"home.dest"
# 
# object type is generic type in pandas that is stored as string.
# 1. Name: Name of the passenger
# 2. Sex: Male or female passenger
# 3. Ticket: Ticket number
# 4. Cabin: cabin number
# 5. Boat: lifeboat
# 6. Embarked: Origin place of passenger
# 7. home.dest: Destination place of passenger
# 
# Numeric type is represented as int with no decimals. Floats are with decimals.
# 1. Pclass : Passenger class
# 2. Survived: 1 - Survived or 0 - died.
# 3. Age: passenger age
# 4. sibsp: spouse or siblings on board.
# 5. parch: No. of parents/children aboard
# 6. fare: fare for travel
# 7. body: body recovered?

# # 2.1 Exploratory data analysis

# In[227]:


df.pclass.value_counts(dropna=False)


# In[228]:


df['survived'].value_counts(dropna=False)


# In[229]:


df['name'].value_counts(dropna=False).head()


# In[230]:


df['sex'].value_counts(dropna=False).head()


# In[231]:


df['age'].value_counts(dropna=False).head()


# In[232]:


df['sibsp'].value_counts(dropna=False).head()


# In[233]:


df['parch'].value_counts(dropna=False).head()


# In[234]:


df['ticket'].value_counts(dropna=False).head()


# In[235]:


df['fare'].value_counts(dropna=False).head()


# In[236]:


df['cabin'].value_counts(dropna=False).head()


# In[237]:


df['embarked'].value_counts(dropna=False).head()


# In[238]:


df['boat'].value_counts(dropna=False).head()


# In[239]:


df['body'].value_counts(dropna=False).head()


# In[240]:


df['home.dest'].value_counts(dropna=False).head()


# # 2.2 Summary statistics

# In[241]:


df.describe()


# # 2.3 Visual exploratory data analysis
# helps in greatway to spot outliers and obvious errors

# In[242]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().magic('matplotlib inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
import seaborn as sns


# In[243]:


sns.set_style('whitegrid')
fig, axis = plt.subplots(figsize=(8,8))
sns.barplot(x='pclass', y='survived', hue='sex', data=df, ax=axis, ci=None)
axis.set_title('Survival Rate by Passenger Class and Gender')
loc, labels = plt.xticks()
plt.xticks(loc, ['First Class','Second Class', 'Third Class'])
axis.set_ylabel('Percent Survived')
axis.set_xlabel('')


# First class female survived most and in general female survived most.

# In[244]:


df.boxplot(column='age', by='survived')


# there are few outliers in age variable.

# In[245]:


df.boxplot(column='fare', by='survived', )


# There are outliers in 'fare' variable also.

# In[246]:


df.plot(kind='scatter', x='age', y='fare', rot=70)
plt.show()


# In[247]:


g = sns.FacetGrid(df, col='survived')
g.map(plt.hist, 'age', bins=20)


# Age group between 15 to 30 are more in survived people. however it is unclear whether more women or men are there in this group. hence we will run one more histogram as below.

# In[248]:


g = sns.FacetGrid(df, col='survived')
g.map(plt.hist, 'sex', bins=20)


# In[249]:


corr = df.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# There is clear correlation between fare and survived rate and number of sibling and family members aboard.

# # 3. Data preparation
# cleaning bad data by converting datatypes for analysis, filling missing data, dropping duplicate data

# In[250]:


df['sex']=df['sex'].astype('category')
df['ticket']=df['ticket'].astype('category')
df['cabin']=df['cabin'].astype('category')
df['embarked']=df['embarked'].astype('category')
df['home.dest']=df['home.dest'].astype('category')
df['boat']=pd.to_numeric(df['boat'],errors='coerce')


# In[251]:


df.info()


# In[252]:


#dropping few variables wchich are insignificant.
df = df.drop(['name','ticket','cabin','body','home.dest'], axis=1)


# In[253]:


df.info()


# In[254]:


#drop duplicates
df=df.drop_duplicates()


# In[255]:


df.info()


# In[256]:


# Tabular view of Data , the type of data , the missing values , unique counts , % Missing}
# Creating the Data Dictionary with first column being datatype.
Data_dict = pd.DataFrame(df.dtypes)
# Identifying unique values . For this I've used nunique() which returns unique elements in the object.
Data_dict['UniqueVal'] = df.nunique()
# Identifying the missing values from the dataset.
Data_dict['MissingVal'] = df.isnull().sum()
# Percentage of Missing Values
Data_dict['Percent Missing'] = round(df.isnull().sum()/len(df)*100, 2)
# identifying count of the variable.
Data_dict['Count'] = df.count()
# Renaming the first column using rename()
Data_dict = Data_dict.rename(columns = {0:'DataType'})
Data_dict


# In[257]:


df[df.embarked.isnull()]


# In[258]:


# Fill missing values with test statistic i.e Mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df[df.embarked.isnull()]


# In[259]:


df[df.fare.isnull()]


# In[260]:


#Fill missing values with median.
df["fare"] = df["fare"].fillna(value=df["fare"].median())
df[df.fare.isnull()]


# In[261]:


#check for missing values and validate Validate Mean and Median
df[df.age.isnull()].head()
df.age.mean() , df.age.median()


# In[262]:


#median and mean are very near. fill the missing values and check.
df["age"] = df["age"].fillna(value=df["age"].median())
df[df.age.isnull()].head()


# In[263]:


import numpy as np
guess_ages = np.zeros((2,3))
guess_ages


# In[264]:


df.info()


# In[265]:


df.drop([1309,]).tail()


# In[266]:


#find sex of passengers for dummy coding.
print("count of passengers")
df.groupby('sex').size()


# In[267]:


#convert (Male, female) to (0,1)
df['sex'] = df.sex.apply(lambda x: 0 if x == "female" else 1)


# In[268]:


#find variable embarked of passengers for dummy coding.
print("count of embarked")
df.groupby('embarked').size()


# In[269]:


#convert (C,Q,S) to (1,2,3)
coding_embarked = {"embarked":{"C": 1, "Q": 2,"S": 3}}
df.replace(coding_embarked, inplace=True)
df.head()


# In[270]:


#for the sake of retaining boat variable, im filling missing value with 0.

df[['boat']]= df[['boat']].fillna(0)


# In[271]:


df.head()


# In[272]:


df.info()


# In[273]:


df['sex']=df['sex'].astype('category')
df['embarked']=df['embarked'].astype('category')
df['boat']=pd.to_numeric(df['boat'],errors='coerce')


# In[274]:


df.info()


# In[279]:


def top_code(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])

for df in [df]:
    df['age'] = top_code(df, 'age', 68)
    df['fare'] = top_code(df, 'fare', 101)


# In[280]:


#asserting the missing values treatment by writing to file and checking before outlier treatment.
df.to_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic_rev02.csv')


# # 4. Modeling

# In[281]:


#load the data
df.to_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic_rev02.csv')


# In[283]:


# Load the dataset again
modeling_df = pd.read_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic.csv')


# In[285]:


import random as rnd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


# In[286]:


from sklearn import metrics
from sklearn.metrics import roc_curve,auc,recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import confusion_matrix,average_precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Ensemble Models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import xgboost as xgb


# In[294]:


# Create data set to train data
x = modeling_df[modeling_df.loc[:, modeling_df.columns != 'survived'].columns]
y = modeling_df['survived']
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=.2, random_state=1)


# In[295]:


x.head()
y.head()
x_train.head()
x_test.head()
y_train.head()
y_test.head()


# In[296]:


# Instntiate , Fit and Predict Test Set
LR = LogisticRegression()
LR.fit(x_train,y_train)
predictionLR=LR.predict(x_test)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictionLR,y_test))


# In[297]:


# Model Evluation : Accuracy Test
print(metrics.classification_report(predictionLR,y_test))


# In[298]:


# Model Evluation :Confusion Matrix
plt.figure(figsize=(5,5))
print('CONFUSION MATRIX')
cfLR=metrics.confusion_matrix(predictionLR,y_test)
sns.heatmap(cfLR , annot = True , cmap = "Greens" , fmt = 'd' )
plt.show()


# In[301]:


from sklearn.metrics import roc_auc_score
LR_roc_auc = roc_auc_score(y_test, LR.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, LR.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % LR_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[302]:


predictionLR_Train=LR.predict(x_train)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictionLR_Train,y_train))
# Model Evluation : Accuracy
print(metrics.classification_report(predictionLR_Train,y_train))


# In[303]:


#LR MODEL
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
models = [LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB() , LinearDiscriminantAnalysis() , 
        QuadraticDiscriminantAnalysis(),XGBClassifier(), Perceptron()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB', 'LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis',
            'XGBoost','Perceptron']

accuracy = []
precision = []
recall = []
roc_auc = []
for model in range(len(models)):
    clf = models[model]
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    accuracy.append(accuracy_score(pred , y_test))
    precision.append(precision_score(pred , y_test))
    recall.append(recall_score(pred , y_test))
     
compare = pd.DataFrame({'Algorithm': model_names , 'Accuracy': accuracy , 'Precision': precision , 'Recall': recall })
compare

