from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


classification_grid_parameters = {
    SVC():  {
        'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    # RandomForestClassifier():   {
    #     'n_estimators': [10, 40, 70, 100],
    #     'max_depth': [3, 5, 7],
    #     'min_samples_split': [0.2, 0.5, 0.7, 2],
    #     'min_samples_leaf': [0.2, 0.5, 1, 2],
    #     'max_features': [0.2, 0.5, 1, 2],
    # },
    KNeighborsClassifier(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPClassifier():    {
        'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500]
    }
}


