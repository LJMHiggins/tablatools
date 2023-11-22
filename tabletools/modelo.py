### Model kit ###
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class MyTunedClassifier:
    """
    A class for constructing a baseline classification model.

    ...

    Attributes
    ----------
    classifier : scikit-learn classification model object
        algorith to use for classification task
    X_train : numpy array
        training dataset
    y_train : numpy array
        training dataset (class labels)
    random_cv_params : dict
        paramaters for random grid search
    cv_parms : dict
        parameters for complete gridsearch
    
    Methods
    -------
    get_best_random_estimator():
        Performs randomised gridsearch using random_cv_params.
    get_best_grid_cv():
        Performs gridsearch using cv_params.
    train_base():
        Trains out of the box baseline model.
    check_fitted(clf):
        Verifies if model has been fit.
    evaluate_model(model, test_features, test_labels):
        Evaluates model of choice against test data.
    """
    def __init__(self, classifier, X_train, y_train, random_cv_params = None, cv_params = None): # As types are mutable, None can be used as place holder
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.random_cv_params = random_cv_params
        self.cv_params = cv_params
        
    def get_best_random_estimator(self):
        if isinstance(self.classifier, RandomForestClassifier):
            random_model = RandomizedSearchCV(estimator = self.classifier, 
                                              param_distributions = self.random_cv_params, 
                                              n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1,
                                              scoring="accuracy")
        
        elif isinstance(self.classifier, LogisticRegression):
            # Investigate logistic regression parameters
            random_model = RandomizedSearchCV(estimator = self.classifier, 
                                              param_distributions = self.random_cv_params, 
                                              n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1,
                                              scoring="accuracy")
        else:
            print("Model is not currently supported by class")
            
        random_model.fit(self.X_train, self.y_train)
        self.random_cv = random_model
        self.best_random = random_model.best_estimator_
        
    def get_best_grid_cv(self):
        grid_search = GridSearchCV(estimator = self.classifier, param_grid = self.cv_params, 
                                   cv = 3, n_jobs = -1, verbose = 2,
                                   scoring="accuracy")
        grid_search.fit(self.X_train, self.y_train)
        self.grid_cv = grid_search
        self.best_grid = grid_search.best_estimator_
        
    def train_base(self):
        base_model = self.classifier
        base_model.fit(self.X_train, self.y_train)
        self.base_model = base_model
        
    def check_fitted(self, clf): 
        return hasattr(clf, "classes_")
    
    def evaluate_model(self, model, test_features, test_labels):
                predictions = model.predict(test_features)
                accuracy = accuracy_score(test_labels, predictions)
                print('Model Performance')
                print('Accuracy = {:0.2f}%.'.format(accuracy))   
                return accuracy 
            
    def evaluate_model_v_base(self, vs_random = True):
        if vs_random == True:
            mod_to_compare = self.best_random
        else: 
            mod_to_compare = self.best_grid
            
        if self.check_fitted(self.base_model) & self.check_fitted(mod_to_compare):
            base_accuracy = self.evaluate_model(self.base_model,
                                                test_features = self.X_test, test_labels = self.y_test)
            model_accuracy = self.evaluate_model(mod_to_compare, 
                                                 test_features = self.X_train, test_labels = self.y_train)
            print('Improvement of {:0.2f}%.'.format(100 * (model_accuracy - base_accuracy) / base_accuracy))
        else:
            print("Model(s) still requires fitting")