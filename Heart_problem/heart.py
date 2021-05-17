import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart.csv')
'''
dataset.head()

dataset.describe()

dataset['output'].value_counts()


figure = plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(),annot=True)

sns.countplot(data=dataset,x=dataset['output'])

plt.figure(figsize = (14, 8),  dpi = 200)
ax = sns.countplot(data = dataset, x = 'age', )
ax.set(ylim = (0, 20))
plt.yticks(np.arange(0, 20))
'''
X = dataset.drop(columns=['age', 'sex'])
y = dataset['output']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

'''
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''
'''
from sklearn.svm import SVC
clf = SVC(C=100,gamma=0.0001,kernel = 'rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200,random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

classification_report(y_test,y_pred)

output = pd.DataFrame({'Actully' : y_test,'Predicted_Output': y_pred})
output.to_csv('submission.csv', index=True)

'''
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
grid_search.fit(X_train, y_train)

best_accuracies = grid_search.best_score_
best_parameters = grid_search.best_params_
'''
'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
'''


'''
from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
gird_search = GridSearchCV(estimator = clf,
                           param_grid = param_grid,
                           scoring = "accuracy",
                           cv = 5,
                           n_jobs =-1)
# fitting the model for grid search
gird_search.fit(X_train, y_train)

grid_predictions = gird_search.predict(X_test)

best_accuracies = gird_search.best_score_
best_parameters = gird_search.best_params_
'''