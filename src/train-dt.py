import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='param-unik', repo_name='mlflow-daghubs-demo', mlflow=True)

# add tracking uri for mlflow 
mlflow.set_tracking_uri("https://dagshub.com/param-unik/mlflow-daghubs-demo.mlflow")

# load Iris datasets 
iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into train and test 
X_train, X_test , y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)

# define the parameters for the random forest classifier
max_depth = 5


# apply mlflow
mlflow.set_experiment(experiment_name='iris-dt')
with mlflow.start_run():
    
    dt = DecisionTreeClassifier(max_depth=max_depth)
    
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    
    accuracy = accuracy_score(y_test ,y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    #create confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', xticklabels=iris.target_names, yticklabels = iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel("Predicted")
    plt.title('Confusion Matrix')
    
    # save the plot as artifact 
    plt.savefig("confusion_matrix.png")
    
    # mlflow code for artifact 
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log the code 
    mlflow.log_artifact(__file__)
    
    # log the model 
    mlflow.sklearn.log_model(dt, 'decision_tree')
    
    # add tags for easy navigation 
    mlflow.set_tag('author', 'param.unik')
    mlflow.set_tag('model', 'decision tree experiment')
        
    print('accuracy -> ', accuracy)
    