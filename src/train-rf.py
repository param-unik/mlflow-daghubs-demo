import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load Iris datasets 
iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into train and test 
X_train, X_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)

# define the parameters for the random forest classifier
max_depth = 15
n_estimators = 10

# apply mlflow
with mlflow.start_run():
    
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators = n_estimators)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test ,y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    print('accuracy -> ', accuracy)
    