import os
import tqdm

from utils import get_audio_config
from data_extractor import load_data
from sklearn.model_selection import GridSearchCV
from utils import extract_feature, get_best_estimators, AVAILABLE_EMOTIONS
from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv
from sklearn.metrics import accuracy_score, make_scorer, mean_absolute_error, mean_squared_error


class EmotionRecognizer:
    def __init__(self, model=None, **kwargs):
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        
        self._verify_emotions()

        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)

        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True

        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name", "tess_ravdess.csv")
        self.emodb_name = kwargs.get("emodb_name", "emodb.csv")
        self.custom_db_name = kwargs.get("custom_db_name", "custom.csv")

        self.verbose = kwargs.get("verbose", 1)

        self._set_metadata_filenames()

        self.write_csv()

        self.data_loaded = False
        self.model_trained = False

        if not model:
            self.determine_best_model()
        else:
            self.model = model

    def _verify_emotions(self):
        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."
    
    def get_best_estimators(self):
        return get_best_estimators(self.classification)

    def load_data(self):
        if not self.data_loaded:
            result = load_data(self.train_desc_files, self.test_desc_files, self.audio_config, self.classification,
                                emotions=self.emotions, balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    def _set_metadata_filenames(self):
        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")
        if self.emodb:
            train_desc_files.append(f"train_{self.emodb_name}")
            test_desc_files.append(f"test_{self.emodb_name}")
        if self.custom_db:
            train_desc_files.append(f"train_{self.custom_db_name}")
            test_desc_files.append(f"test_{self.custom_db_name}")

        self.train_desc_files = train_desc_files
        self.test_desc_files  = test_desc_files

    def write_csv(self):
        for train_csv_file, test_csv_file in zip(self.train_desc_files, self.test_desc_files):
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                if not self.override_csv:
                    continue
            if self.emodb_name in train_csv_file:
                write_emodb_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated EMO-DB CSV File")
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated TESS & RAVDESS DB CSV File")
            elif self.custom_db_name in train_csv_file:
                write_custom_csv(emotions=self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated Custom DB CSV File")

    def grid_search(self, params, n_jobs=2, verbose=1):
        score = accuracy_score if self.classification else mean_absolute_error
        
        print("Grid Search starts!")
        # GridSearchCV - Exhaustive search over specified parameter values for an estimator.
        grid = GridSearchCV(estimator=self.model, param_grid=params, scoring=make_scorer(score),
                            n_jobs=n_jobs, verbose=verbose, cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)
        print("Grid Search over!")
        return grid_result.best_estimator_, grid_result.best_params_, grid_result.best_score_

    def determine_best_model(self):
        if not self.data_loaded:
            self.load_data()
        
        estimators = self.get_best_estimators()

        result = []

        if self.verbose:
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(f"Evaluating {estimator.__class__.__name__}")
            
            detector = EmotionRecognizer(estimator, emotions=self.emotions, tess_ravdess=self.tess_ravdess,
                                        emodb=self.emodb, custom_db=self.custom_db, classification=self.classification,
                                        features=self.features, balance=self.balance, override_csv=False)
            # data already loaded
            detector.X_train = self.X_train
            detector.X_test  = self.X_test
            detector.y_train = self.y_train
            detector.y_test  = self.y_test
            detector.data_loaded = True
            
            detector.train(verbose=0)
            accuracy = detector.test_score()
            result.append((detector.model, accuracy))

        # sort the result
        # regression: best is the lower, not the higher
        # classification: best is higher, not the lower
        result = sorted(result, key=lambda item: item[1], reverse=self.classification)
        best_estimator = result[0][0]
        accuracy = result[0][1]
        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification:
                print(f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy*100:.3f}% test accuracy")
            else:
                print(f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error")

    def train(self, verbose=1):
        if not self.data_loaded:
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]
    
    def predict_proba(self, audio_path):
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")


    def test_score(self):
        y_pred = self.model.predict(self.X_test)
        if self.classification:
            return accuracy_score(y_true=self.y_test, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_test, y_pred=y_pred)


    def train_score(self):
        y_pred = self.model.predict(self.X_train)
        if self.classification:
            return accuracy_score(y_true=self.y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_train, y_pred=y_pred)

