from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# refining train data
data = pd.read_csv("train.csv")
data = data[["Survived","Pclass", "Sex", "Age", "SibSp","Parch", "Fare", "Embarked"]]
data['Sex'] = np.where(data['Sex']=='male', 1, 0)
data['Fare'] = data['Fare'].replace(0, data['Fare'].mean())
data['Age'] = data['Age'].replace(np.NaN, data['Age'].mean())
data['Age'] = np.log(data['Age'])
data['Fare'] = np.log(data['Fare'])
data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2})

# refining with test data
test_data = pd.read_csv('test.csv')
test_data = test_data[["Pclass", "Sex", "Age", "SibSp","Parch", "Fare", "Embarked"]]
test_data['Sex'] = np.where(test_data['Sex']=='male', 1, 0)
test_data['Fare'] = test_data['Fare'].replace(0, test_data['Fare'].mean())
test_data['Age'] = test_data['Age'].replace(np.NaN, test_data['Age'].mean())
test_data['Age'] = np.log(test_data['Age'])
test_data['Fare'] = np.log(test_data['Fare'])
test_data['Embarked'] = test_data['Embarked'].map({'S':0,'C':1,'Q':2})

# getting hold of train data without a surviving status
x = data.drop(['Survived'],axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=0)

# defining parameters
params = {
    'n_estimators': range(6, 11),
    'max_depth': range(3, 8),
    'learning_rate': [.2, .3, .4],
    'colsample_bytree': [.7, .8, .9, 1]
}

# predicting survival for test data
Xgb = XGBClassifier()
g_search = GridSearchCV(estimator = Xgb, param_grid = params,
    cv= 3, n_jobs = 1, verbose = 0, return_train_score=True)
g_search.fit(X_train, y_train)
predictions = g_search.predict(test_data)

final_submission = pd.read_csv('test.csv')
final_submission['Survived'] = np.array(predictions)
final_submission[['PassengerId', 'Survived']].to_csv('final.csv', index=False)

