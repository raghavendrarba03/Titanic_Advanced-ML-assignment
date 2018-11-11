# Titanic_Advanced-ML-assignment
#survival of passengers aboard RNS Titanic.

import pandas as pd

#Import file

df=pd.read_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\titanic.csv')

#2. Data Understanding

df.head()

df.tail()

df.columns

df.shape

df.info()


#2.1 Exploratory data analysis

df.pclass.value_counts(dropna=False)

df['survived'].value_counts(dropna=False)

df['name'].value_counts(dropna=False).head()

df['sex'].value_counts(dropna=False).head()

df['age'].value_counts(dropna=False).head()

df['sibsp'].value_counts(dropna=False).head()

df['parch'].value_counts(dropna=False).head()

df['ticket'].value_counts(dropna=False).head()

df['fare'].value_counts(dropna=False).head()

df['cabin'].value_counts(dropna=False).head()

df['embarked'].value_counts(dropna=False).head()

df['boat'].value_counts(dropna=False).head()

df['body'].value_counts(dropna=False).head()

df['home.dest'].value_counts(dropna=False).head()

#2.2 Summary statistics

df.describe()

#2.3 Visual exploratory data analysis

import matplotlib.pyplot as plt
%matplotlib inline
get_ipython().magic('matplotlib inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
import seaborn as sns

sns.set_style('whitegrid')
fig, axis = plt.subplots(figsize=(8,8))
sns.barplot(x='pclass', y='survived', hue='sex', data=df, ax=axis, ci=None)
axis.set_title('Survival Rate by Passenger Class and Gender')
loc, labels = plt.xticks()
plt.xticks(loc, ['First Class','Second Class', 'Third Class'])
axis.set_ylabel('Percent Survived')
axis.set_xlabel('')

df.boxplot(column='age', by='survived')

df.boxplot(column='fare', by='survived', )

df.plot(kind='scatter', x='age', y='fare', rot=70)
plt.show()

g = sns.FacetGrid(df, col='survived')
g.map(plt.hist, 'age', bins=20)

g = sns.FacetGrid(df, col='survived')
g.map(plt.hist, 'sex', bins=20)

corr = df.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()

#3. Data preparation

df['sex']=df['sex'].astype('category')
df['ticket']=df['ticket'].astype('category')
df['cabin']=df['cabin'].astype('category')
df['embarked']=df['embarked'].astype('category')
df['home.dest']=df['home.dest'].astype('category')
df['boat']=pd.to_numeric(df['boat'],errors='coerce')

df.info()

#dropping few variables wchich are insignificant.
df = df.drop(['name','ticket','cabin','body','home.dest'], axis=1)

df.info()

#drop duplicates
df=df.drop_duplicates()

df.info()

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

df[df.embarked.isnull()]

# Fill missing values with test statistic i.e Mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df[df.embarked.isnull()]

df[df.fare.isnull()]

#Fill missing values with median.
df["fare"] = df["fare"].fillna(value=df["fare"].median())
df[df.fare.isnull()]

#check for missing values and validate Validate Mean and Median
df[df.age.isnull()].head()
df.age.mean() , df.age.median()

#median and mean are very near. fill the missing values and check.
df["age"] = df["age"].fillna(value=df["age"].median())
df[df.age.isnull()].head()

import numpy as np
guess_ages = np.zeros((2,3))
guess_ages

df.info()

df.drop([1309,]).tail()

#find sex of passengers for dummy coding.
print("count of passengers")
df.groupby('sex').size()

#convert (Male, female) to (0,1)
df['sex'] = df.sex.apply(lambda x: 0 if x == "female" else 1)

#find variable embarked of passengers for dummy coding.
print("count of embarked")
df.groupby('embarked').size()

#convert (C,Q,S) to (1,2,3)
coding_embarked = {"embarked":{"C": 1, "Q": 2,"S": 3}}
df.replace(coding_embarked, inplace=True)
df.head()

#for the sake of retaining boat variable, im filling missing value with 0.
df[['boat']]= df[['boat']].fillna(0)

df.head()

df.info()

df['sex']=df['sex'].astype('category')
df['embarked']=df['embarked'].astype('category')
df['boat']=pd.to_numeric(df['boat'],errors='coerce')

def top_code(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])

for df in [df]:
    df['age'] = top_code(df, 'age', 68)
    df['fare'] = top_code(df, 'fare', 101)
    
#asserting the missing values treatment by writing to file and checking before outlier treatment.
df.to_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic_rev02.csv')

# 4. Modeling

#load the data
df.to_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic_rev02.csv')

# Load the dataset again
modeling_df = pd.read_csv('C:\\Users\\Raghavendra Reddy\\Desktop\\Titanic assignment\\prepared_titanic.csv')

import random as rnd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

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

# Create data set to train data
x = modeling_df[modeling_df.loc[:, modeling_df.columns != 'survived'].columns]
y = modeling_df['survived']
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=.2, random_state=1)

x.head()
y.head()
x_train.head()
x_test.head()
y_train.head()
y_test.head()

# Instntiate , Fit and Predict Test Set
LR = LogisticRegression()
LR.fit(x_train,y_train)
predictionLR=LR.predict(x_test)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictionLR,y_test))

# Model Evluation : Accuracy Test
print(metrics.classification_report(predictionLR,y_test))

# Model Evluation :Confusion Matrix
plt.figure(figsize=(5,5))
print('CONFUSION MATRIX')
cfLR=metrics.confusion_matrix(predictionLR,y_test)
sns.heatmap(cfLR , annot = True , cmap = "Greens" , fmt = 'd' )
plt.show()

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

predictionLR_Train=LR.predict(x_train)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictionLR_Train,y_train))
# Model Evluation : Accuracy
print(metrics.classification_report(predictionLR_Train,y_train))

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
