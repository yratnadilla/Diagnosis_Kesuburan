import numpy as np
import pandas as pd

df = pd.read_csv('fertility.csv')
# print(df.isnull().sum())
# print(df.head())

dfX = df.drop(['Season', 'Diagnosis'], axis= 'columns')
dfY = df['Diagnosis']
# print(dfX.head())
# print(dfY.head())

# =============================================================
# one hot encoding
import category_encoders as ce
ohe = ce.OneHotEncoder(use_cat_names= True)
dfX = ohe.fit_transform(dfX)
# print(dfX.columns.values)
    
# =============================================================
# splitting data
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dfX,
    dfY,
    test_size = 0.1,
    random_state= 1
)

# =============================================================
# decision tree
from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier()
modelDT.fit(xtr, ytr)

# =============================================================
# extreme random forest
from sklearn.ensemble import ExtraTreesClassifier
modelERF = ExtraTreesClassifier(n_estimators=20)
modelERF.fit(xtr, ytr)

# =============================================================
# k-nearest neighbors
def nNeighbors():
    x = round(len(dfX) ** 0.5)
    if x % 2 == 0:
        return x + 1
    else:
        return x

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=nNeighbors())
modelKNN.fit(xtr, ytr)

# =============================================================
# logistic regression
from sklearn.linear_model import LogisticRegression
modelLOG = LogisticRegression(solver='liblinear')
modelLOG.fit(xtr, ytr)

# =============================================================
# cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

k = StratifiedKFold(n_splits=100)

# print(
#     'Decision Tree CV score : ',
#     cross_val_score(
#     DecisionTreeClassifier(),
#     xtr,
#     ytr,
#     cv=5
# ).mean())

# print(
#     'Extreme Random Forest CV score : ',
#     cross_val_score(
#     ExtraTreesClassifier(n_estimators=20),
#     xtr,
#     ytr,
#     cv=5
# ).mean())

# print(
#     'K-Nearest Neighbors CV score : ',
#     cross_val_score(
#     KNeighborsClassifier(n_neighbors=nNeighbors()),
#     xtr,
#     ytr,
#     cv=5
# ).mean())

# print(
#     'Logistic Regression CV score : ',
#     cross_val_score(
#     LogisticRegression(solver='liblinear'),
#     xtr,
#     ytr,
#     cv=5
# ).mean())

# =============================================================
# prediction

dfPredict = pd.read_csv('patients.csv')

dfName = dfPredict['Name']
dfPredictX = dfPredict.drop(['Name'], axis='columns')

# print(dfName)
# print(dfPredictX)

for i in range(5):
    yPredDT = modelDT.predict([dfPredictX.iloc[i]])
    print(dfName.iloc[i],'- fertility prediction based on Decision Tree: ', yPredDT[0])

    ypredERF = modelERF.predict([dfPredictX.iloc[i]])
    print(dfName.iloc[i],'- fertility prediction based on Extreme Random Forest: ', yPredDT[0])

    ypredKNN = modelKNN.predict([dfPredictX.iloc[i]])
    print(dfName.iloc[i],'- fertility prediction based on K-Nearest Neighbors: ', yPredDT[0])

    ypredLOG = modelLOG.predict([dfPredictX.iloc[i]])
    print(dfName.iloc[i],'- fertility prediction based on Logistic Regression: ', yPredDT[0])

