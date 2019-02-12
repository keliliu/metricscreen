import numpy as np
from copy import deepcopy
from .utils import _logit
from .utils import _expit


class ClassificationEnsembler:
    """Object to ensemble together the binary classification models used for residual updates in
    a metric path.
    """

    def __init__(self, class_proportion, pi_bound):
        """
            Parameters
            ----------
            class_proportion : float
                The proportion of class 1 instances in the training set.
            pi_bound : float
                Stepsize parameter used for metric path. The predicted probabilities from each weak learner
                (for the training data) are restricted to lie between 1 - pi_bound and pi_bound.
        """

        assert 0 < class_proportion < 1, "class_proportion should be a float between 0 and 1."

        self.class_proportion = class_proportion
        self.model_list = []  # a sequential list of the models used for residual updating along a metric path
        self.training_vars = []  # for each model in model_list, the indices of the variables used for training
        self.all_chosen_vars = np.array([], dtype=np.int)
        self.bound_list = []  # list of truncation points used in metric path
        self.eta_bound = np.abs(_logit(pi_bound))

    def add_model(self, model, var_ids):
        """Adds a new model to the ensemble.

            Parameters
            ----------
            model : object
                A trained model used for residual update along metric path.
            var_ids : integer array
                Indices of variables used to train model.
        """

        self.model_list += [deepcopy(model)]
        self.training_vars += [deepcopy(var_ids)]
        self.all_chosen_vars = np.union1d(self.all_chosen_vars, np.array(var_ids, dtype=np.int))

    def extend_model(self, model, X, y):
        """ Extends the current ensemble of models with a new model.

            Parameters
            ----------
            model : model object
                model should be equipped with the fit and predict_proba methods.In addition, the arguments
                to fit should include option for sample_weight.
            X : array like, shape = (n,p)
                Feature matrix.
            y : array like, shape = (n,)
                Binary response vector.
        """

        pi = self.predict_proba(X=X)[:, 1]
        w = y * (1-pi) + (1-y) * pi
        w = w/np.sum(w)

        model.fit(X=X[:, self.all_chosen_vars], y=y, sample_weight=w)
        self.add_model(model, self.all_chosen_vars)
        self.add_bound(0)

    def remove_model_extension(self):
        """ Removes extension to the ensemble of models. Deletes the final element of model_list, training_vars, and bound_list.
        """
        self.model_list.pop()
        self.training_vars.pop()
        self.bound_list.pop()

    def add_bound(self, bound):
        """
            Parameters
            ----------
            bound : float
                For the training sample the maximum absolute logit score.
        """
        self.bound_list += [bound]

    def predict_proba(self, X):
        """Fits estimated class probabilities by ensembling together weak learners. 
        
            Parameters
            ----------
            X : array like, shape = (n,p)
                Feature matrix for dataset that we want to predict for.
            
            Returns
            -------
            proba : array like, shape = (n,2)
                Matrix with two columns containing the predicted class probabilities. First column contains the
                predicted probabilities for class 0.
        """

        n, p = X.shape
        pi = np.ones(n) * self.class_proportion

        for k in range(len(self.model_list)):
            theta = self.model_list[k].predict_proba(X[:, self.training_vars[k]])[:, 1]
            logit_theta = _logit(theta)

            if self.bound_list[k] != 0:
                logit_theta = logit_theta/self.bound_list[k] * self.eta_bound
                theta = _expit(logit_theta)

            a = pi * theta
            b = (1-pi) * (1-theta)
            pi = a/(a + b)

        return np.vstack([1-pi, pi]).T

    def predict(self, X):
        return 1*(self.predict_proba(X=X)[:, 1] > 0.5)
