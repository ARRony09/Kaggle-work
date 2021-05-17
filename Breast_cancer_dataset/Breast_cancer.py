import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset=pd.read_csv('dataR2.csv')

X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values

dataset.describe()

classes = dataset['Classification']
sns.countplot(x=classes,data=dataset)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset['Classification']=label_encoder.fit_transform(dataset['Classification'])

sns.heatmap(dataset.corr(),annot=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''
'''

classifier = LinearDiscriminantAnalysis(solver='lsqr')
classifier.fit(X_train, y_train.ravel())
y_pred = classifier.predict(X_test)
'''
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
'''

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300,random_state=0)
classifier.fit(X_train,y_train)

# Applying gird_search
from sklearn.model_selection import GridSearchCV

parameters= {
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

gird_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 5,
                           n_jobs =-1)
gird_search = gird_search.fit(X_train,y_train)

best_accuracies = gird_search.best_score_
best_parameters = gird_search.best_params_

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies.mean()


'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
'''
'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Applying gird_search
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100],'kernel':['linear']},
              {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
gird_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs =-1)
gird_search = gird_search.fit(X_train,y_train)

best_accuracies = gird_search.best_score_
best_parameters = gird_search.best_params_
'''
'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train.ravel())
dtc_pred = classifier.predict(X_test)

from sklearn.model_selection import GridSearchCV
criterions = ['gini', 'entropy']
parameters = dict(criterion=criterions)
dtc = GridSearchCV(
    classifier, parameters, cv=5, scoring='accuracy'
)
aaccuracy = 0.58
'''

classifier.score(X_test,y_test)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)


