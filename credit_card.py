import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#read the data
data = pd.read_csv('creditcard.csv')
print(data.columns)
print(data.shape)

print(data.describe())

data = data.sample(frac=0.1, random_state=1)     #to describe fraction of data

print(data.shape)

data.hist(figsize=(25, 25)) #plot to histogram
plt.show()

#calculating the fraud and valid data
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_function= len(Fraud)/float(len(Valid))
print(outlier_function)

print('Fraud cases {}'.format(len(Fraud)))
print('Valid cases {}'.format(len(Valid)))

corrmat= data.corr()
fig = plt.figure(figsize=(12, 9))
sb.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

#get all columns from dataframe
columns = data.columns.tolist()  #to take out the columns from the columns

#to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

#store the variable we will predict on
target = "Class"

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.describe)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest #isolation- randomly selects the feature and random split value b/w min and max
from sklearn.neighbors import LocalOutlierFactor #localoutlier= calculates the score of anamoly

#define a random state
state=1

#define outlier direction methods
classifiers = {
    "Isolation forest ": IsolationForest(max_samples=len(X),contamination=outlier_function,random_state=state),
    "Local_outlier ": LocalOutlierFactor(n_neighbors=20, contamination=outlier_function)
}

#fit the model

n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    #fit the outliers and tag outliers
    if clf_name == "Local_outlier ":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_

    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

        #reshape value 0 to valid and 1 to valid
    y_pred[y_pred==1] =0
    y_pred[y_pred==-1] =1

    n_error = (y_pred !=Y).sum()

    #run classification metric
    print('{}: {}'.format(clf_name,n_error))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y, y_pred))


