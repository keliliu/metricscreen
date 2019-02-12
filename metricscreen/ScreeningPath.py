import numpy as np
from .model_ensembler import ClassificationEnsembler
from .utils import _expit
from .utils import _logit


class VarTracker:
    """
    Maintains a dictionary that counts the number of times variables have been selected.
    """

    def __init__(self):
        self.tracker = dict()
        self.ranker = dict()
        self.worst_rank = 0

    def update(self, new_vars):

        old_vars = np.array([*self.tracker.keys()], dtype=np.int)
        only_in_old = old_vars[np.invert(np.isin(old_vars, new_vars))]

        # update count of number of times a variable is seen
        for var in new_vars:
            self.tracker[var] = self.tracker.get(var, 0) + 1

        # update the rank of a variable (smaller rank = more important)
        for rr in range(len(new_vars)):
            self.ranker[new_vars[rr]] = self.ranker.get(new_vars[rr], self.worst_rank) + rr

        for var in only_in_old:
            self.ranker[var] = self.ranker[var] + len(new_vars)

        self.worst_rank += len(new_vars)

    def filter(self, cutoff):
        return dict((k, v) for k, v in self.tracker.items() if v > cutoff)

    def filter_and_rank(self, cutoff):
        filtered = np.array([*self.filter(cutoff=cutoff)], dtype=np.int)
        rankings = np.zeros(len(filtered))

        for ii in range(len(rankings)):
            rankings[ii] = self.ranker[filtered[ii]]

        return filtered[np.argsort(rankings)]


class MetricPath:
    """
    Manages training of multiple metric paths. Each path comprises of successive rounds of screening followed
    by residual update.
    """

    def __init__(self,
                 num_tries=1,
                 num_seen_cutoff=0,
                 target_num_var=np.Infinity,
                 max_rounds=50,
                 penalty_decrease_factor=0.95,
                 min_times_on_path=1,
                 take_top_k=None,
                 weight_update='balancing',
                 pi_hat_bound=0.52,
                 subsample_feature=1,
                 subsample_data=1
                 ):
        """
            Parameters
            ----------
            num_tries : int, (default = 1)
                In each step of the screening path, how many times should the screening method be rerun? When trying to
                detect pure interactions, it may be necessary to rerun multiple times to find interaction (especially
                in high dimensions).
            num_seen_cutoff : int, (default = 0)
                If num_tries > 1, how many times does a variable have to be chosen by the screening method before
                we record it as having been 'selected'? Can be used to ensure that we only return variables that
                are consistently selected (i.e. belong to stationary points with large basins of attraction).
            target_num_var : int, (default = Infinity)
                Stop path early once target_num_var variables have been selected.
            max_rounds : int, (default = 50)
                Maximum number of screen/residual rounds to take within a single screening path. Path terminates early
                if target_num_var variables have been selected before max_rounds is reached.
            penalty_decrease_factor : float, (default = 0.95)
                If a particular screening run selects no variables and target_num_var has not been reached,
                then the penalty value is decreased by penalty_decrease_factor (assuming it is less than 1).
            min_times_on_path : int, (default = 1)
                Number of times that a variable needs to be chosen by screener on a single path in order for it to be
                counted as 'chosen' for that path. Alternate way to promote stable solutions when num_seen_cutoff
                is set to 0.
            take_top_k : int, (default = None)
                In each screening step, only the top k chosen variables (largest coefficients) are recorded,
                the remaining are ignored. By default, all chosen variables will be recorded.
            weight_update : string, (default = 'balancing')
                Specifies the type of weight update, either 'balancing' or 'adaboost'.
            pi_hat_bound : float, (default = 0.52)
                Both 'balancing' and 'adaboost' updates alter the current sample weight based on probability
                estimates from a weak learner. The estimated probabilities from the weak learner are truncated
                above at pi_hat_bound and below at 1 - pi_hat_bound. This truncation controls the learning rate
                along the path. Generally, setting pi_hat_bound close to 0.5 yields better performance. The cost
                is that a greater number of rounds need be performed to select the same number of variables.
            subsample_feature : float, (default = 1)
                Specifies the fraction of features to subsample in each round of a screening path.
            subsample_data : float, (default = 1)
                Specifies the fraction of data to subsample for each screening path. Note that subsampling of
                data happens a single time at the beginning of the screening path. There is no subsampling of data
                across rounds of a screening path.
        """

        self.num_tries = num_tries
        self.num_seen_cutoff = num_seen_cutoff
        self.max_rounds = max_rounds
        self.target_num_var = target_num_var
        self.penalty_decrease_factor = penalty_decrease_factor
        self.min_times_on_path = min_times_on_path
        self.take_top_k = take_top_k

        self.weight_update = weight_update
        self.pi_hat_bound = pi_hat_bound
        self.subsample_feature = subsample_feature
        self.subsample_data = subsample_data

        self.times_in_path = VarTracker()

    def _update_tracker(self, new_vars):
        self.times_in_path.update(new_vars)

    def get_selected_vars(self, times_seen_cutoff=1):
        return self.times_in_path.filter(times_seen_cutoff)

    def get_selected_vars_as_array(self, times_seen_cutoff=1):
        # return variables that were selected strictly greater than the cutoff number of times
        return np.sort(np.array([*self.get_selected_vars(times_seen_cutoff=times_seen_cutoff)], dtype=np.int))

    def get_ranked_var_list(self, times_seen_cutoff=1):
        return self.times_in_path.filter_and_rank(cutoff=times_seen_cutoff)

    def reset_var_tracker(self):
        self.times_in_path = VarTracker()

    def run_paths(self,
                  num_paths,
                  screener,
                  model,
                  X,
                  class_p,
                  sample_weight=None,
                  lam=0.0,
                  build_model=True,
                  verbose=True,
                  **kwargs):
        """Run metric path for num_paths number of times returning a list of the variables
        selected by each path. Automatically tracks the number of times each variable has
        been selected.

            Parameters
            ----------
            num_paths : int
                Number of paths to train.
            screener : object of class Metric Learner
                Screening object must have method train.
            model : model object
                Model for performing residual steps along a metric path. Usually a classifier from scikit-learn.
                The model object must be equipped with the methods fit and predict_proba.
            X : array like, shape = (n,p)
                Feature matrix.
            class_p : array like, shape = (n,)
                Binary response vector (coded 0 or 1). The algorithm also allows for soft class labels in which
                case the ith element of class_p denotes the probability that sample i belongs to class 1.
            sample_weight : array like, optional, shape = (n,)
                Weights for each of the samples. By default, if no weights are supplied, the algorithm will
                normalize the sample_weight vector so that class 0 and class 1 observations have the same total weight.
            lam : float, (default = 0.0)
                l1-penalty to use for metric screening.
            build_model : boolean, (default = True)
                While screening for variables along a metric path, the algorithm can simultaneously create
                (in a boosting like fashion) an ensemble model from the weak learners used in the residual update step.
            verbose : boolean, (default = True)
                Whether to print intermediate output such as the variables selected in each round of training.
            kwargs: dictionary
                Keyword arguments to the train method of the user supplied screening object.

            Returns
            -------
            chosen_vars : list of lists
                Each element of the list is another list containing the variables selected for a single path.
        """

        chosen_vars = []
        ensemble_list = []
        for num in range(num_paths):
            screener.reset()  # reset MetricLearner before starting next path
            var_ids, ensemble = _learn_metric_path(screener=screener,
                                                   model=model,
                                                   num_tries=self.num_tries,
                                                   num_seen_cutoff=self.num_seen_cutoff,
                                                   target_num_var=self.target_num_var,
                                                   max_rounds=self.max_rounds,
                                                   penalty_decrease_factor=self.penalty_decrease_factor,
                                                   min_times_on_path=self.min_times_on_path,
                                                   take_top_k=self.take_top_k,
                                                   weight_update=self.weight_update,
                                                   pi_hat_bound=self.pi_hat_bound,
                                                   subsample_feature=self.subsample_feature,
                                                   subsample_data=self.subsample_data,
                                                   X=X,
                                                   class_p=class_p,
                                                   sample_weight=sample_weight,
                                                   lam=lam,
                                                   build_model=build_model,
                                                   verbose=verbose,
                                                   **kwargs)
            chosen_vars.append(var_ids)
            ensemble_list.append(ensemble)
            self._update_tracker(chosen_vars[num])

        if not build_model:
            ensemble_list = None

        return chosen_vars, ensemble_list


def _multiple_try_screener(screener, num_tries, num_seen_cutoff=0, **kwargs):
    """ Repeated calls the train method of the user provided screening object and
    returns the variables that are found > cutoff number of times.

        Parameters
        ----------
        screener : object of class MetricLearner
            The screening object must have a train method.
        num_tries : int
            Number of times to call the train method of the user supplied screener.
        num_seen_cutoff : int, (default = 0)
            Only variables that are selected more than num_seen_cutoff times will be reported.

        Returns
        -------
        var_list : array-like, shape = (m,)
            Indices of variables that are selected more than num_seen_cutoff times.
    """

    var_tracker = VarTracker()
    for num in range(num_tries):
        if not kwargs.get('warm_start'):
            screener.reset_beta()  # reset beta between tries (warm_starts tend to get stuck)
        screener.train(**kwargs)
        var_tracker.update(screener.get_vars())

    return np.array([*var_tracker.filter(num_seen_cutoff).keys()]).astype(np.int)


def _update_balancing_weight(y, phat, w):
    """
    Computes balancing weight update from response vector and predicted probabilities.
    Weight gets multiplied by 1-phat if y = 1 and multiplied by phat if y = 0 (thereby
    upweighting incorrectly classified observations).
    :param y: Binary numpy array of responses. Values are either 0 or 1.
    :param phat: Numpy array of same length as y giving P(Y=1|x).
    :param w: Numpy array of same length as y. Weight vector to be updated.
    :return: Updated weight vector
    """
    u = y * (1 - phat) + (1 - y) * phat
    w = w * u
    return w / np.sum(w)


def _update_adaboost_weight(y, phat, w, learning_rate=1):
    """
    Computes adaboost weight update from response vector and predicted probabilities.
    :param y: Binary numpy array of responses. Values are either 0 or 1.
    :param phat: Numpy array of same length as y giving P(Y=1|x).
    :param w: Numpy array of same length as y. Weight vector to be updated.
    :param learning_rate: Learning rate for adaboost.
    :return: Updated weight vector
    """

    ptrunc = np.clip(phat, a_min=1e-8, a_max=1 - 1e-8)  # truncate for numerical stability

    u = (-0.5 * (2 * y - 1) * np.log(ptrunc / (1 - ptrunc))) * learning_rate
    u = u - np.max(u)  # for stability subtract off the max, otherwise exponential function may explode
    w = w * np.exp(u)
    return w / np.sum(w)


def _shrink_pi_hat(pi_hat, pi_hat_bound):
    eta_hat = _logit(pi_hat)
    eta_bound = np.abs(_logit(pi_hat_bound))
    eta_max = np.max(np.abs(eta_hat))

    if eta_max < eta_bound:
        return pi_hat, 0
    else:
        eta_hat = (eta_hat / np.max(np.abs(eta_hat))) * eta_bound
        return _expit(eta_hat), eta_max


def _update_weight(y, phat, w, weight_update):
    """
    Wrapper function to call either balancing weight update or adaboost weight udate.
    """

    if weight_update == 'adaboost':
        return _update_adaboost_weight(y=y, phat=phat, w=w)
    elif weight_update == 'balancing':
        return _update_balancing_weight(y=y, phat=phat, w=w)
    else:
        raise ValueError("Unrecognized choice of weight update.")


def _define_subgroup(sampled_features, chosen_var):
    """
    Returns the indices of the chosen variables with respect to the set of sampled features.
    In the case of an empty intersection, returns None.
    """
    # no variables selected so far
    if len(chosen_var) == 0:
        return None

    # no variables selected among the sampled features
    if sampled_features is None:
        return chosen_var

    return np.where(np.isin(sampled_features, chosen_var))[0]


def _subsample_features(X, feature_fraction, always_include=np.array([], dtype=np.int)):
    """
    Subsamples a fraction of the columns of X.
    :param X: feature matrix
    :param feature_fraction: fraction of indices to sample
    :param always_include: column indices of features that should always be included
    :return: Xsub which is the subsampled matrix, sampled_features which includes the indices of the sampled columns
    """

    assert feature_fraction > 0, "Fraction of features to sample must be > 0."

    if feature_fraction == 1:
        return X, None
    else:
        num_sampled_features = int(np.ceil(feature_fraction * X.shape[1]))
        sampled_features = np.random.choice(X.shape[1], size=num_sampled_features, replace=False)
        sampled_features = np.union1d(sampled_features, always_include)
        return X[:, sampled_features], sampled_features


def _get_top_k(var_ids, scores, take_top_k=None):
    if take_top_k is None:
        return var_ids

    return var_ids[np.argsort(-scores)][:np.min([take_top_k, len(var_ids)])]


def _update_var_list(sorted_list, var_ids, scores, take_top_k=None):
    """Updates a sorted list of variables with potentially new variables
    contained in var_ids. Order for new variables is determined by scores
    (new variables with higher scores are added first). Only top_k variables
    in var_ids is considered.

        Parameters
        ----------
        sorted_list : array like, type = np.int
            Indices (in ranked order) of the selected variables.
        var_ids : array like, type = np.int
            Variables to be considered for inclusion into sorted_list. This can include
            variables already in sorted_list as well as new variables.
        scores : array like
            Scores with which we should order the variables in var_ids.
        take_top_k : int, optional, (default = None)
            Only the top_k variables in var_ids are considered for inclusion into sorted_list. By
            default, all variables are considered.

        Returns
        -------
        sorted_list : array like, type = np.int
            Updated sorted list of selected variables.
    """
    if len(var_ids) == 0:
        return sorted_list

    var_ids = var_ids[np.argsort(-scores)]  # rank by scores
    if take_top_k is not None:
        var_ids = var_ids[:np.min([take_top_k, len(var_ids)])]
    new_bool = np.invert(np.isin(var_ids, sorted_list))
    new_vars = var_ids[new_bool]

    print("New variables: ", new_vars)

    sorted_list = np.concatenate([sorted_list, new_vars])

    return sorted_list


def _learn_metric_path(screener,
                       model,
                       num_tries,
                       num_seen_cutoff,
                       target_num_var,
                       max_rounds,
                       penalty_decrease_factor,
                       min_times_on_path,
                       take_top_k,
                       weight_update,
                       pi_hat_bound,
                       subsample_feature,
                       subsample_data,
                       X,
                       class_p,
                       sample_weight,
                       lam,
                       build_model=True,
                       verbose=True,
                       **kwargs):
    """ Implements training of metric path by iterating between two operations:
        1. Variable selection using metric screening.
        2. Model fitting using the selected variables and residualing out their effects using balancing
        or adaboost weight update.

        Parameters
        ----------
        Please see __init__ and run_path methods for specification of the input parameters.

        Returns
        -------
        selected : array like (type = int)
            Indices of selected variables
        ensemble : object of class ClassificationEnsembler
            The sequence of weak learners that were used to perform the residual updates.
    """
    var_counter = VarTracker()  # keeps track of how many times a variable has been selected on path
    selected = np.array([], dtype=np.int)

    ensemble = ClassificationEnsembler(class_proportion=np.mean(class_p), pi_bound=pi_hat_bound)

    # if hierarchical:
    #     assert init_value is not None, "If hierarchical search, initial value for active set coefficients must be set."
    #     beta_init = 'zero'
    #     subgroup_lower = init_value
    #     subgroup_upper = init_value*hier_upper

    subgroup_var = None

    if subsample_data < 1:
        sub_id = np.random.choice(X.shape[0], size=int(np.ceil(subsample_data * X.shape[0])), replace=False)
        X = X[sub_id, :]
        class_p = class_p[sub_id]
        if sample_weight is not None:
            sample_weight = sample_weight[sub_id]

    if sample_weight is None:
        # By default each class is normalized to have the same total weight
        w = class_p * 1 / np.sum(class_p) + (1 - class_p) * 1 / np.sum(1 - class_p)
        w = w / np.sum(w)
    else:
        assert len(sample_weight) == X.shape[0], "Sample weight vector must be same dimension as number of rows of X."
        w = sample_weight

    # subsample variables
    Xsub, sampled_features = _subsample_features(X, subsample_feature)

    # run the supplied screening algorithm
    var_id = _multiple_try_screener(screener=screener,
                                    num_tries=num_tries,
                                    num_seen_cutoff=num_seen_cutoff,
                                    X=Xsub,
                                    class_p=class_p,
                                    sample_weight=w,
                                    subgroup=subgroup_var,
                                    lam=lam,
                                    **kwargs)

    if verbose:
        print("Metric screening coefficients: ", np.round(-np.sort(-screener.beta[var_id]), decimals=2))

    # keep track of chosen variables as an integer array
    chosen_var = np.array([], dtype=np.int)
    scores = screener.beta[var_id]
    if subsample_feature < 1:
        var_id = sampled_features[var_id]
    top_k_vars = _get_top_k(var_ids=var_id, scores=scores, take_top_k=take_top_k)
    var_counter.update(top_k_vars)
    chosen_var = _update_var_list(sorted_list=chosen_var, var_ids=var_id, scores=scores, take_top_k=take_top_k)

    if verbose:
        print("Total of %d variables selected so far. Variables selected in round %d: "
              % (len(chosen_var), 0), var_id[np.argsort(-scores)], flush=True)

    # terminate if no variables are chosen and no sub-sampling of features is performed
    if len(var_id) == 0 and subsample_feature == 1:
        if target_num_var < np.Infinity:
            lam *= penalty_decrease_factor  # decrease penalty
            print("Decreasing value of lambda to %f because target number of variables has not been reached." % lam)
        else:
            return chosen_var, ensemble

    round_num = 1
    while round_num < max_rounds:

        # Reweight the sample
        if len(chosen_var) > 0:
            if len(top_k_vars) > 0:
                pi_hat = model.fit(X=X[:, top_k_vars], y=class_p, sample_weight=w).predict_proba(X[:, top_k_vars])[:, 1]
                if build_model:
                    ensemble.add_model(model=model, var_ids=top_k_vars)
            else:
                pi_hat = model.fit(X=X[:, chosen_var], y=class_p, sample_weight=w).predict_proba(X[:, chosen_var])[:, 1]
                if build_model:
                    ensemble.add_model(model=model, var_ids=chosen_var)

            pi_hat, eta_max = _shrink_pi_hat(pi_hat=pi_hat, pi_hat_bound=pi_hat_bound)
            if build_model:
                ensemble.add_bound(eta_max)

            if verbose:
                print("Smallest fitted probability: ", np.round(np.min(pi_hat), decimals=2))
                print("Largest fitted probability: ", np.round(np.max(pi_hat), decimals=2))

            w = _update_weight(y=class_p, phat=pi_hat, w=w, weight_update=weight_update)

        # subsample features (always include the features that have already been chosen)
        Xsub, sampled_features = _subsample_features(X, subsample_feature, always_include=chosen_var)

        # if hierarchical:
        #     # update subgroup (index with respect to sampled features)
        #     subgroup_var = _define_subgroup(sampled_features, chosen_var)
        #     beta_init = np.zeros(X.shape[1]) if subsample_feature == 1 else np.zeros(len(sampled_features))
        #     beta_init[subgroup_var] = init_value

        var_id = _multiple_try_screener(screener=screener,
                                        num_tries=num_tries,
                                        num_seen_cutoff=num_seen_cutoff,
                                        X=Xsub,
                                        class_p=class_p,
                                        sample_weight=w,
                                        lam=lam)

        if verbose:
            print("Metric screening coefficients: ", np.round(-np.sort(-screener.beta[var_id]), decimals=2),
                  flush=True)

        # terminate if hierarchical and all weights remain at initial values
        # if hierarchical and (np.sum(screener.beta) < (np.sum(beta_init) + 1e-4)):
        #     break

        # terminate if no variable chosen and no sub-sampling has been performed
        if len(var_id) == 0 and subsample_feature == 1:
            if target_num_var < np.Infinity:
                lam *= penalty_decrease_factor  # decrease penalty
                print("Decreasing value of lambda to %f because target number of variables has not been reached." % lam)
            else:
                break

        scores = screener.beta[var_id]
        if subsample_feature < 1:
            var_id = sampled_features[var_id]
        top_k_vars = _get_top_k(var_ids=var_id, scores=scores, take_top_k=take_top_k)
        var_counter.update(top_k_vars)
        chosen_var = _update_var_list(sorted_list=chosen_var, var_ids=var_id, scores=scores, take_top_k=take_top_k)

        if verbose: print("Total of %d variables selected so far. Variables selected in round %d: " \
                          % (len(chosen_var), round_num), var_id[np.argsort(-scores)], flush=True)

        round_num += 1

        selected = np.array([*var_counter.filter(cutoff=np.min([max_rounds - 1, min_times_on_path]))], dtype=np.int)
        if len(selected) >= target_num_var:
            if not build_model:
                break
            # before exiting, need to add variables found in final round to ensemble
            if len(top_k_vars) > 0:
                pi_hat = model.fit(X=X[:, top_k_vars], y=class_p, sample_weight=w).predict_proba(X[:, top_k_vars])[:, 1]
                ensemble.add_model(model=model, var_ids=top_k_vars)
            else:
                pi_hat = model.fit(X=X[:, chosen_var], y=class_p, sample_weight=w).predict_proba(X[:, chosen_var])[:, 1]
                ensemble.add_model(model=model, var_ids=chosen_var)

            pi_hat, eta_max = _shrink_pi_hat(pi_hat=pi_hat, pi_hat_bound=pi_hat_bound)
            ensemble.add_bound(eta_max)
            break

    print("Variables selected without filtering: ", chosen_var)
    print("Variables selected after filtering: ", selected, flush=True)

    return selected, ensemble
