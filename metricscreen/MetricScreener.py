import numpy as np
import numbers
from . import random_sampler
from scipy.stats import rankdata
from .projections import *


class MetricLearner:
    
    def __init__(self,
                 rank_transform=False,
                 l1_constraint=1.0,
                 linf_constraint=1.0,
                 subgroup_l1=np.Infinity,
                 subgroup_lower=0,
                 subgroup_upper=1,
                 f=lambda u: np.exp(-u),
                 g=lambda u: np.exp(-u),
                 diff=np.abs,
                 stepsize=1.0,
                 beta_tol=1e-3,
                 solver='dual average',
                 param_update_frac=1,
                 batch_size=500,
                 num_rounds_before_check=25,
                 active_set_stable_threshold=2):

        """Screener that learns metric appropriate for classification.

            Parameters
            ----------
            rank_transform : boolean
                Boolean for whether to rank transform each column of feature matrix before learning.
            l1_constraint : float (default = 1.0)
                l-1 bound on the coefficient vector.
            linf_constraint : float (default = 1.0)
                l-infinity bound on the coefficient vector.
            subgroup_l1 : float (default = np.Infinity)
                Bound on sum of coefficients of features in designated subgroup (used only when subgroup of variables
                is specified during training). CURRENTLY NOT USED.
            subgroup_lower : float, optional (default = 0.0)
                Lower bound on coefficients of features in designated subgroup (used only when subgroup of variables
                is specified during training).
            subgroup_upper : float, optional (default = 1.0)
                Upper bound on coefficients of features in designated subgroup (used only when subgroup of variables
                is specified during training).
            f : function, (default = exp(-x))
                The f-function in kernel screening.
            g : function, (default = exp(-x))
                The g-function in kernel screening (negative derivative of the f-function).
            diff : function (default = absolute value)
                Function that is applied to the pairwise differences before applying the f-function.
            stepsize : float, (default = 1.0)
                Stepsize to use for sgd or dual averaging.
            beta_tol : float (default = 1e-3)
                Variables with coefficient values smaller than this threshold are ignored when computing
                the pairwise difference between pairs.
            solver : string, (default = 'dual average')
                Optimization routine that is used. Either 'dual average' or 'sgd'.
            param_update_frac : frac, (default = 1.0)
                Fraction of parameters to update when using steepest descent algorithm.
            batch_size : int, (default = 150)
                Batch size used when estimating the gradient.
            num_rounds_before_check : int (default = 25)
                Number of training rounds before each check for convergence.
            active_set_stable_threshold : int (default = 25)
                The active set of coefficient vector needs to remain the same for this number of convergence
                checks before declaring convergence.
        """

        self.solver = solver

        self.rank_transform = rank_transform

        assert f is not None, "Must specify similarity/dissimilarity function, f."
        self.f = f
        assert g is not None, "Must specify g = -f' for gradient computations."
        self.g = g

        self.diff = np.abs if diff is None else diff

        self.l1_constraint = l1_constraint
        self.linf_constraint = linf_constraint
        self.subgroup = None
        self.subgroup_lower = subgroup_lower
        self.subgroup_upper = subgroup_upper
        self.subgroup_l1 = subgroup_l1

        self.lam = None
        self.beta_tol = beta_tol  # coefficients smaller than this threshold are considered 0

        self.iters = 1
        self.batch_size = batch_size
        self.stepsize = stepsize
        self.param_update_frac = param_update_frac
        self.num_rounds_before_check = num_rounds_before_check
        self.active_set_stable_threshold = active_set_stable_threshold
        self.active_set_stable = 0  # counter for how many rounds the active set has been stable

        self.n = None
        self.p = None
        self.X = None

        # private variables for keeping track of observation weights and sampling observations according to weight
        self.sample_weight = None  # weight given to each sample point
        self.class_p = None  # conditional probability that a particular observation belongs to class 1
        self.sum_class0 = None  # total sample weight of class 0
        self.sum_class1 = None  # total sample weight of class 1
        self.pairs_frac0 = None  # probability of drawing a 0,0 pair given a draw of concordant observations
        self.p0 = None  # prob of drawing an observation given that a class 0 observation is drawn
        self.p1 = None  # prob of drawing an observation given that a class 1 observation is drawn
        self.sampler0 = None  # RandomSampler for drawing from class 0 according to distribution p0
        self.sampler1 = None  # RandomSampler for drawing from class 1 according to distribution p1

        # kernel weights and dual variables used in dual averaging
        self.beta = None
        self.dual_var = None
        self.old_beta = None
        self.active = None


    def set_X(self, X):
        if self.rank_transform:
            self.X = np.apply_along_axis(rankdata, 0, X) / (self.n + 1)
        else:
            self.X = X

    def set_penalty(self, lam):
        if lam is None:
            self.lam = 0.5*np.sqrt(np.log(self.p)/self.n)
        else:
            assert isinstance(lam, numbers.Number), "Penalty parameter must be a real number."
            self.lam = lam

    def init_dual(self, init_value):
        if type(init_value) == np.ndarray:
            self.dual_var = init_value.reshape((self.p,))
        elif isinstance(init_value, numbers.Number):
            self.dual_var = np.ones(self.p)*init_value
        else:
            raise ValueError("Unrecognized initial value for dual variable in dual averaging.")

    def init_beta(self, init_option, init_frac_of_l1=0.8):

        """Called by train to initialize coefficient vector (kernel weights).

            Parameters
            ----------
            init_option : string or array-like
                If array-like, then the initial value of the coefficient vector. Else a string equal to
                either 'zero', 'random', or 'equal'. For 'random' and 'equal', the sum of the initial coefficients is
                set to init_frac_of_l1 * l1_constraint.
            init_frac_of_l1: float, (default = 0.8)
                Initial l1-norm of coefficient vector is set to this fraction of the l1 constraint. Only used for
                'random' initialization and 'equal' initialization.
        """

        if type(init_option) == np.ndarray:
            self.beta = init_option.reshape((self.p,))
            return

        assert init_option in set(('random', 'equal', 'zero')), "Options for initialization are random, equal, or zero."

        if init_option == 'zero':
            self.beta = np.zeros(self.p)
        elif init_option == 'random':
            self.beta = np.min([self.l1_constraint, 1])*np.random.dirichlet(np.ones(self.p)*0.25, size=1).squeeze()*init_frac_of_l1
        elif init_option == 'equal':
            self.beta = np.min([self.l1_constraint, 1])*np.ones(self.p)/self.p*init_frac_of_l1
        else:
            raise ValueError("Unrecognized option for initializing beta.")
          
    def _compute_g(self, D, beta, active):
        # only compute g using features with coefficient > tol
        if len(active) == 0:
            return np.ones(D.shape[0])*self.g(0)
        else:
            return self.g(np.matmul(D[:, active], beta[active]))

    def _compute_f(self, D, beta, active):
        return self.f(np.matmul(D[:, active], beta[active]))

    def eval_objective(self):

        d_discord = self.sample_diff(self.X, self.X, self.sampler0, self.sampler1)
        d_concord0 = self.sample_diff(self.X, self.X, self.sampler0, self.sampler0)
        d_concord1 = self.sample_diff(self.X, self.X, self.sampler1, self.sampler1)

        active = self.beta >= self.beta_tol
        f_discord = np.average(self._compute_f(d_discord, self.beta, active))
        f_concord0 = np.average(self._compute_f(d_concord0, self.beta, active))
        f_concord1 = np.average(self._compute_f(d_concord1, self.beta, active))

        return self.pairs_frac0*f_concord0 + (1-self.pairs_frac0)*f_concord1 - f_discord

    def _is_converged(self, old, new):

        """ Performs check of whether the support of the coefficient vector has converged.
        Ignores the magnitudes of the non-zero values of the coefficient vector as we only
        care about which variables are selected. Declares convergence if the active set
        has been stable for > active_set_stable_threshold checks where each check is performed
        after num_rounds_before_check rounds of training.
        """

        old_active = (old >= self.beta_tol)
        new_active = (new >= self.beta_tol)

        num_old = np.sum(old_active)
        num_new = np.sum(new_active)

        overlap = np.sum(old_active * new_active)/np.max([num_old, num_new, 1])

        if overlap == 1:
            self.active_set_stable += 1

        if self.active_set_stable > self.active_set_stable_threshold:
            return True
        else:
            return False

    def sample_pairs(self, sampler0, sampler1):

        """Sample pairs (i,j) with replacement with probability proportional to w0_i*w0_j

            Parameters
            ----------
            num_pairs: Integer. Number of pairs to sample.
            sampler0: RandomSampler
            sampler1: RandomSampler

            Returns
            -------
            pair1 : 1d array, shape = (m,)
                Indices for the first observation in the sampled pairs.
            pair2 : 1d array, shape = (m,)
                Indices for the second observation in the sampled pairs.
        """

        pair1 = sampler0.next()
        pair2 = sampler1.next()

        return pair1, pair2

    def sample_diff(self, arr0, arr1, sampler0, sampler1):
        """
        Randomly sample squared differences diff(arr0[i,:] - arr1[j,:]) from 2-D arrays arr0 and arr1
        according to row weights specified by RandomSamplers sampler0 and sampler1. The diff function
        is specified when initializing a MetricLearner instance and defaults to absolute value function.
        :param arr0: 2-D array
        :param arr1: 2-D array
        :param sampler0: RandomSampler
        :param sampler1: RandomSampler
        """

        n0, p0 = arr0.shape
        n1, p1 = arr1.shape

        assert p0 == p1, "A and B must same column dimension."

        pair0, pair1 = self.sample_pairs(sampler0, sampler1)
        return self.diff(arr0[pair0, :] - arr1[pair1, :])

    def get_stochastic_grad(self, X, beta, active, sampler0, sampler1, lam):

        # between group difference
        d_discord = self.sample_diff(arr0=X, arr1=X, sampler0=sampler0, sampler1=sampler1)
        g_discord = self._compute_g(D=d_discord, beta=beta, active=active).reshape((-1,1))
        grad_discord = np.average(d_discord*g_discord, axis=0)  # d_discord * g_discord uses numpy broadcasting

        # within group difference for class 0
        d_concord0 = self.sample_diff(arr0=X, arr1=X, sampler0=sampler0, sampler1=sampler0)
        g_concord0 = self._compute_g(D=d_concord0, beta=beta, active=active).reshape((-1,1))
        grad_concord0 = np.average(d_concord0*g_concord0, axis=0)

        # within group difference for class 1
        d_concord1 = self.sample_diff(arr0=X, arr1=X, sampler0=sampler1, sampler1=sampler1)
        g_concord1 = self._compute_g(D=d_concord1, beta=beta, active=active).reshape((-1,1))
        grad_concord1 = np.average(d_concord1*g_concord1, axis=0)

        return self.pairs_frac0*grad_concord0 + (1-self.pairs_frac0)*grad_concord1 - grad_discord + lam

    def get_vars(self):
        return np.where(self.beta >= self.beta_tol)[0]

    def init_sample_weight(self, sample_weight=None):
        """
        Each observation can be weighted differently. By default, uniform weighting.
        :param sample_weight: Numpy array with same length as number of observations.
        All entries must be positive. Entries not required to sum to 1.
        """
        if sample_weight is None:
            self.sample_weight = np.ones(self.n) / self.n
        else:
            assert len(sample_weight) == self.n, "Length of sample weight must be same as number of observations."
            self.sample_weight = sample_weight

    def init_response_weight(self, class_p):
        """
        Responses are allowed to be "soft" i.e. probabilities P(Y=1|x) rather than binary (0-1)
        values. The probability that the ith observation is sampled and its response is 1 is
        proportional to

        sample_weight[i] * class_p[i].

        Similarly the probability that ith observation is sampled and its response is 0 is proportional to

        sample_weight[i] * (1-class_p[i]).

        These two quantities can be normalized to give the probability of sampling an observation
        conditional on being in class 1 or class 0 respectively.
        :param class_p: Numpy array with each entry between 0 and 1. Each try represents P(Y=1|x).

        """
        assert 0 <= np.min(class_p) and np.max(class_p) <= 1, "Probabilities must be between 0 and 1."
        assert self.sample_weight is not None, "Initialize sample weights first before response weight."

        # class_p contains probability that observation i has response equal to 1
        self.class_p = class_p
        self.sum_class1 = np.sum(self.class_p * self.sample_weight)  # total weight of class 1
        self.sum_class0 = np.sum((1 - self.class_p) * self.sample_weight)  # total weight of class 0

        # probability of drawing a within class pair in which both responses are 0 (other
        # possibility is that both responses are 1)
        self.pairs_frac0 = self.sum_class0 ** 2 / (self.sum_class0 ** 2 + self.sum_class1 ** 2)

        # probability of drawing observation given it is in class 0
        self.p0 = (1 - self.class_p) * self.sample_weight / self.sum_class0
        self.sampler0 = random_sampler.RandomSampler(self.n, batch_size=self.batch_size, weight=self.p0)

        # probability of drawing observation given it is in class 1
        self.p1 = self.class_p * self.sample_weight / self.sum_class1
        self.sampler1 = random_sampler.RandomSampler(self.n, batch_size=self.batch_size, weight=self.p1)

    def reset(self):
        """
        Call reset after performing screening along a path and before starting new path.
        """
        self.iters = 1
        self.beta = None
        self.old_beta = None
        self.dual_var = None
        self.sample_weight = None
        self.subgroup = None

    def reset_beta(self):
        self.beta = None
        self.old_beta = None

    def train(self,
              X,
              class_p,
              sample_weight=None,
              subgroup=None,
              max_iter=5000,
              beta_init='equal',
              dual_init=0.0,
              lam=None,
              verbose=False,
              zero_tol=1e-6,
              warm_start=False,
              warmup=100):

        """Performs training of metric learner.

            Parameters
            ----------
            X : array like, shape = (n,p)
                feature matrix
            class_p: array like, shape = (n,)
                Typically a binary response vector. Algorithm also allows for 'fuzzy' responses in which case
                the ith element of class_p denotes the probability that the ith response is in class 1.
            sample_weight: array like, shape = (n,), optional
                Weights for the data instances. Defaults to equal weights for all instances.
            subgroup: array like, optional, (default = None)
                Indices for subgroup of features to be singled out for special attention.
            max_iter : int, (default = 5000)
                Maximum number of steps to take in optimization routine.
            beta_init : string or array like, (default = 'random')
                Specifies conditions for initialization of coefficient vector. If string, then either
                'random', 'zero', or 'equal'. Otherwise beta_init is an array giving the initial coefficient
                vector.
            dual_init : float or array like, (default = 0.0)
                Initial value for dual variable in dual averaging algorithm (only used when solver is dual averaging).
                If float, then all dual variables are initialized at given value.
            lam : float or None
                l1 penalty. If None, then a theoretical value is chosen.
            zero_tol : float, (default = 1e-6)
                If l1 norm of coefficient vector falls below this threshold (after warmup rounds are completed),
                then we declare that the coefficient vector is 0.
            verbose : bool, (default = False)
                Whether interim output should be printed during training.
            warm_start : bool (default = False)
                Whether to begin training with current value of coefficient vector (if it is available).
            warmup : int, (default = 100)
                Only check convergence after warmup rounds have completed.


        """
        self.n, self.p = X.shape
        self.set_X(X)

        self.init_sample_weight(sample_weight=sample_weight)
        self.init_response_weight(class_p=class_p)

        self.warmup = warmup  # only check convergence after warmup has been satisfied

        # initialize coefficient vector
        if self.beta is None or not warm_start:
            self.init_beta(beta_init)
        else:
            assert len(self.beta) == self.p, "Warm start set to true but length of coefficient vector not equal to\
            number of features in X matrix."
            self.warmup = 25

        self.old_beta = self.beta.copy()
        self.init_dual(init_value=dual_init)

        # initialize active variables
        self.active = np.nonzero(self.beta > 0)[0]

        self.set_penalty(lam)
        self.subgroup = subgroup

        self.active_set_stable = 0
        self.iters = 1

        self.continue_training(max_iter=max_iter, verbose=verbose, zero_tol=zero_tol)

    def _get_steepest_gradients(self, grad):
        cutoff = np.percentile(grad, q=self.param_update_frac*100)
        grad[(grad > cutoff) & (grad < 0)] = 0
        return grad

    def _dual_average_update(self, dual_var, grad, stepsize, iter_count):
        dual_var -= stepsize/np.sqrt(iter_count) * grad
        beta = project_simplex_with_box_constraints(dual_var, l1_norm=self.l1_constraint, linf=self.linf_constraint)
        if self.subgroup is not None:
            beta[self.subgroup] = np.clip(beta[self.subgroup], a_min=self.subgroup_lower, a_max=self.subgroup_upper)
        return beta

    def _sgd_update(self, grad, stepsize, iter_count):
        self.beta -= stepsize/np.sqrt(iter_count) * grad
        self.beta = project_simplex_with_box_constraints(self.beta, l1_norm=self.l1_constraint, linf=self.linf_constraint)

    def continue_training(self, max_iter, verbose=False, zero_tol=1e-6):

        for i in range(max_iter):
            self.active = np.nonzero(self.beta > 0)[0]
            grad = self.get_stochastic_grad(X=self.X, beta=self.beta, active=self.active,
                                            sampler0=self.sampler0, sampler1=self.sampler1, lam=self.lam)

            if self.param_update_frac < 1:
                grad = self._get_steepest_gradients(grad)  # option to update only steepest gradients

            if self.solver == 'dual average':
                self.beta = self._dual_average_update(dual_var=self.dual_var, grad=grad, stepsize=self.stepsize,
                                                      iter_count=self.iters)
            else:
                self._sgd_update(grad=grad, stepsize=self.stepsize, iter_count=self.iters)

            if (i > self.warmup) and (i % self.num_rounds_before_check == 0) \
                    and ((np.sum(np.abs(self.beta)) < zero_tol)
                         or self._is_converged(old=self.old_beta, new=self.beta)):
                break

            if i % self.num_rounds_before_check == 0:
                self.old_beta = self.beta.copy()

            self.iters += 1
            if verbose and (i % 100 == 0):
                print(np.round(self.beta, decimals=2))

        print("Training lasted for %d iterations." % self.iters, flush=True)

        if i == (max_iter-1):
            print('Maximum iteration reached without convergence. Set max_iter higher.', flush=True)


    def hier_train(self, init_value, beta=None, active_set=np.array([], dtype=np.int64), verbose=False, zero_tol=1e-6, \
                   convergence_tol=1e-4, marginal_sampling_size=5000):
        """

        :param active_set: p, boolean array-like
        Boolean array indicating which variables have already been selected at the start of hierarchical training.
        """

        if beta is None: beta = np.zeros(self.p)
        if len(active_set) > 0: beta[active_set] = init_value

        # we use a larger sampling size in the first step to determine whether any gradients are negative
        sampler0 = random_sampler.RandomSampler(self.n, batch_size=marginal_sampling_size, weight=self.p0, buf=3)
        sampler1 = random_sampler.RandomSampler(self.n, batch_size=marginal_sampling_size, weight=self.p1, buf=3)
        grad = self.get_stochastic_grad(X=self.X, beta=beta, active=active_set, sampler0=sampler0, sampler1=sampler1,\
                                        lam=self.lam)
        chosen = np.nonzero(grad < 0)[0]

        # check if any new variables have been added
        chosen = np.union1d(chosen, active_set)
        if len(chosen) == len(active_set):
            return active_set

        return self.hier_train(init_value=init_value, beta=beta, active_set=chosen, verbose=verbose, zero_tol=zero_tol,\
                               convergence_tol=convergence_tol, marginal_sampling_size=marginal_sampling_size)