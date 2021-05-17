import pandas as pd
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

dataset=pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

#dataset.drop(labels=['Unnamed: 0','Clothing ID','Age','Title'],axis=1,inplace=True).reset_index()
from numpy import nan

'''
def is_good(review,recommend):
    if review==nan and recommend==1:
        review='Good'
    elif review==nan and recommend==0:
        review='bad'

dataset['Good']=list(map(is_good,dataset['Review Text'],dataset['Recommended IND']))
'''
'''
sns.heatmap(dataset.corr(),annot=True)
dataset.isnull().sum()
dataset.dropna(subset=['Review Text'],inplace=True)

dataset.isnull().sum()
'''

corpus=[]

#review = dataset[['Review Text', 'Recommended IND']]

#review = review.rename(columns={"Recommended IND": "Recommended", "Review Text": "Review"})

#review = review.dropna(axis=0,how='any')
dataset.dropna(subset=['Review Text'],inplace=True)

dataset=dataset.drop(["Unnamed: 0"],axis=1).reset_index()

review = dataset[['Review Text', 'Recommended IND']]

review = review.rename(columns={"Recommended IND": "Recommended", "Review Text": "Review"})


X=review['Review']
y=review['Recommended']


#X=review['Review'].astype(str)

for i in range(len(X)):
    clean_data = re.sub(r'\W',' ', str(X[i]))
    clean_data = re.sub(r'\d',' ',clean_data)
    clean_data = clean_data.encode('ascii','ignore').decode()
    clean_data = re.sub('[%s]' % re.escape(string.punctuation), ' ', clean_data)
    clean_data = clean_data.lower()
    clean_data = re.sub(r'\s+[a-z]\s+',' ',clean_data)
    clean_data = re.sub(r'^[a-z]\s+',' ',clean_data)
    clean_data = re.sub(r'\s+', ' ', clean_data)
    clean_data = clean_data.split()
    ps = PorterStemmer()
    clean_data = [ps.stem(word) for word in clean_data if not word in set(stopwords.words('english'))]
    clean_data = ' '.join(clean_data)
    #clean_data = clean_data.split()
    #clean_data="".join(i for i in clean_data if ord(i)<128)
    corpus.append(clean_data)
    
    
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer(max_df=0.7)
X = Vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.svm import SVC
clf = SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

'''
# Fitting naive bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
'''
clf.score(X_test,y_test)

y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)


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


from sklearn.metrics import classification_report  
# print classification report
print(classification_report(y_test, grid_predictions))

'''
best_accuracies = grid.best_score_
best_parameters = grid.best_params_
'''
'''
print(grid.best_score_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
'''

