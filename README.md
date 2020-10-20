# Task-4
#importing liabraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()
sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)
0	5.1	3.5	1.4	0.2
1	4.9	3.0	1.4	0.2
2	4.7	3.2	1.3	0.2
3	4.6	3.1	1.5	0.2
4	5.0	3.6	1.4	0.2
iris_df['Species'] = iris.target
iris_df.shape
(150, 5)
iris_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   Species            150 non-null    int32  
dtypes: float64(4), int32(1)
memory usage: 5.3 KB
iris_df.describe()
sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)	Species
count	150.000000	150.000000	150.000000	150.000000	150.000000
mean	5.843333	3.057333	3.758000	1.199333	1.000000
std	0.828066	0.435866	1.765298	0.762238	0.819232
min	4.300000	2.000000	1.000000	0.100000	0.000000
25%	5.100000	2.800000	1.600000	0.300000	0.000000
50%	5.800000	3.000000	4.350000	1.300000	1.000000
75%	6.400000	3.300000	5.100000	1.800000	2.000000
max	7.900000	4.400000	6.900000	2.500000	2.000000
sns.pairplot(iris_df,height = 2.5,diag_kind = "kde" ,hue = 'Species' ,palette = 'Dark2')
plt.figure(figsize = (14, 14))
binsize = 10
plt.subplot(2,2,1)
sns.displot(a = iris_df["petal length (cm)"], bins = binsize)
plt.subplot(2,2,2)
sns.displot(a = iris_df["petal width (cm)"], bins = binsize)
plt.subplot(2,2,3)
sns.displot(a = iris_df["sepal length (cm)"], bins = binsize)
plt.subplot(2,2,4)
sns.displot(a = iris_df["sepal width (cm)"], bins = binsize)
plt.figure(figsize = (10,7))
sns.heatmap(iris_df.corr(), annot = True)
plt.tight_layout()
x = iris_df.drop('Species', axis = 1)
y = iris_df.Species
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
print('Shape of x_train:',x_train.shape)
print('Shape of y_train:',y_train.shape)

print('\n')

print('Shape of x_test:',x_test.shape)
print('Shape of y_test:',y_test.shape)
Shape of x_train: (112, 4)
Shape of y_train: (112,)


Shape of x_test: (38, 4)
Shape of y_test: (38,)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtree = DecisionTreeClassifier(random_state = 4)
dtree.fit(x_train,y_train)
plt.figure(figsize = (19,19))
tree.plot_tree(dtree,filled = True, rounded = True, proportion = True, node_ids = True, feature_names = iris.feature_names)
plt.show()
