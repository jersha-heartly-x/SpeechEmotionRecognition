import pickle

from emotion_recognition import EmotionRecognizer
from parameters import classification_grid_parameters

emotions = ['sad', 'neutral', 'happy']

# number of parallel jobs during the grid search
n_jobs = 4

best_estimators = []

for model, params in classification_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsClassifier":
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation ROC_AUC score!")

print(f"[+] Pickling best classifiers for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))
