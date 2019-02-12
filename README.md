
# Metric Screening Tutorial
This guide will walk you through basic usage for the python package `metricscreen`. There are many screening/variable selection algorithms targeted at linear signals, e.g. the Lasso or t-statistic screening. Metric screening complements such algorithms by focusing on highly nonlinear signals such as pairwise interactions (and higher order interactions). The metric screening algorithm is nonparametric (making no assumptions about how a set of features $X = (X_1, \ldots, X_p)$ is related to the response $Y$). The only restriction is that $Y$ must be binary (though we hope to relax this in the near future!).

## Installation

After cloning the git repository, simply change into the directory containing the file `setup.py` and from the command line run:

```bash
sudo pip install .
```

The package is now ready to use!

## A Simple Example

We'll demonstrate the package's basic functionality on a condensed version of the EEG eye state dataset found on the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State). The full dataset has 14980 observations but we'll work with a random sample of 2000 observations.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import sklearn.ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

The two primary classes in the `metricscreen` package are:
* `MetricLearner`: Performs a single iteration of the metric screening algorithm.
* `MetricPath`: Allows for iterating between metric screening and residual updating. Training a metric path can be thought of as applying a boosting type algorithm to `MetricLearner` (which is a "weak screener").


```python
from metricscreen import MetricLearner
from metricscreen import MetricPath
```

The condensed EEG data is available in the data subdirectory of the package repo.


```python
dat = pd.read_csv('./data/eeg_condensed.csv')
print("Dataset has %d observations and %d features." % (dat.shape[0], dat.shape[1]-1))
dat.head()
```

    Dataset has 2000 observations and 14 features.




```python
x = dat.values[:,1:]
y = dat.values[:,0]
```

The EEG data has 14 continuous features (measurements from electrodes) and one binary response (denoting whether the eye is open or closed). Before performing any analysis, we'll normalize our features by (1) performing a rank transformation, (2) performing a z-transformation, and (3) standardizing the z-transformed features to have mean 0 and variance 1.


```python
def z_transform(x):
    assert type(x) == np.ndarray, "Input should be numpy array."
    return scipy.stats.norm.ppf(np.apply_along_axis(scipy.stats.rankdata, 0, x)/ (x.shape[0] + 1))

x = z_transform(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)
```

To make the data suitable for testing out the variable selection capabilities of our model, let us add some noise variables to the original 14 features. We can add two types of features: those that are completely independent of the original features and those correlated with the original features. We'll begin with 150 independent noise features and 50 correlated noise features.


```python
p_ind_noise = 150
p_cor_noise = 50
rho = 0.75  # level of correlation with the original feature

# adding independent noise
if p_ind_noise > 0:
    x_noise = x[:,np.random.choice(x.shape[1], size=p_ind_noise, replace=True)]
    x_noise = np.hstack([x, np.apply_along_axis(np.random.permutation, axis=0, arr=x_noise)])
else:
    x_noise = x
    
# adding dependent noise
if p_cor_noise > 0:
    x_cor_noise = x[:,np.random.choice(x.shape[1], size=p_cor_noise, replace=True)]
    x_cor_noise = rho*x_cor_noise + np.sqrt(1-np.power(rho,2))*np.random.normal(size=(x_cor_noise.shape[0], p_cor_noise))
    x_noise = np.hstack([x_noise,x_cor_noise])
```

Next, we'll split the data into train and test sets. We'll reserve 75% of the data for testing. This means that our training set will have 500 observations and 214 features (14 signal + 200 noise).


```python
from sklearn.model_selection import train_test_split
xtr, xtes, ytr, ytes = train_test_split(x_noise, y, test_size=0.75, random_state=42)
```

We'll compare the variable selection performance of 4 different methods (you're of course encouraged to augment the code skeleton below with your own favorite screening tools!): (1) no screening (2) t-statistic screening (3) metric screening and (4) oracle screening (if we had prophetic powers and knew which were the 14 original features).


```python
screening_methods = ['no-screening', 't-screen', 'metric-screen', 'oracle']
screener_id = dict()
for k, screener in enumerate(screening_methods): screener_id[screener] = k
    
chosen_var = dict()
for screener in screening_methods: chosen_var[screener] = []

chosen_var['no-screening'] = np.arange(0, xtr.shape[1], 1)
chosen_var['oracle'] = np.arange(0, 14, 1)
```

Next, let's initialize our `MetricLearner` (which acts as the base screener) and `MetricPath` (which combines a base screener with a prediction model that is used to perform residual updates) objects.


```python
# metric learner parameters
l1_bound = 2

# metric path parameters
pi_hat_bound = 0.51
take_top_k = None
target_num_var = 6
max_rounds = 50
min_times_on_path = 1

metric_learner = MetricLearner(l1_constraint=l1_bound)
            
metric_path = MetricPath(take_top_k=take_top_k, pi_hat_bound = pi_hat_bound, target_num_var = target_num_var,\
                         max_rounds=max_rounds, min_times_on_path=min_times_on_path)
```

Let's take a quick look at the input parameters to `MetricLearner` and `MetricPath`.
* `l1_bound`: The metric screening algorithm tries to learn a parametrized metric for feature space. If we let $\beta_j \geq 0$ denote weight given to the $j$th feature, then the `l1_bound` parameter bounds $\sum_j \beta_j$. Smaller values of `l1_bound` allows us to go after smoother signals while larger values are targeted at less smooth signals.

`MetricLearner` allows the user a great deal of flexibility, e.g., the user can specify lower and upper bounds that apply to a subset of variables. The user can also change the similarity/disimilarity function used for metric screening (by default a Laplace kernel is used). You are of course encouraged to read the documentation, but things should still work pretty well even if you just use default settings. For `MetricPath` the important parameters are:

* `pi_hat_bound`: This is analogous to the stepsize parameter in gradient boosting. In each step of the metric path, we perform a residual update by fitting a prediction model using the variables chosen in this step. The fitted probabilities from the model are rescaled so that they all lie between $[1-c,c]$ where $c$ is specified by `pi_hat_bound`. By choosing $c$ close to 0.5, we take smaller steps when constructing our metric path. The default value of `pi_hat_bound` is 0.52 (if computation cost is not a concern, moving this parameter closer to 0.5 may increase performance).
* `max_rounds`: This specifies the maximum number of steps to take for a metric path. Early termination occurs if the target number of variables we are looking for is found before `max_rounds` or if $\beta_j=0$ for all $j$ (indicating no signal above the noise threshold) before `max_rounds`.
* `target_num_var`: Number of variables we want to select. By default it is set to infinity i.e. we will perform `max_rounds` for each path. 
* `take_top_k`: When variables are highly correlated, we may not want to add all variables found in each metric screening step to our set of chosen variables. Instead, it may be effective to only add the top $k$. By default `take_top_k` is set to `None` which means that all variables are automatically added.
* `min_times_on_path`: If a variable only appears a single time on a metric path, it probably isn't very informative (especially if `pi_hat_bound` is close to 0.5 so that we are taking small steps). At the end of the path, we can choose to keep variables that have only appeared `min_times_on_path`. 

Finally, we need to specify the model to be used for residual updating and the $\ell_1$ penalty, `lam`, to be used for metric screening. The model we use for residual updating must be able to handle `sample_weight` during fitting, e.g. we can use sklearn's gradient boosting classifier and random forest classifier. Performance will differ a bit across different choices for `lam` but it is usually okay simply to set `lam` equal to 0 as we do below (as long as your signal to noise ratio is not too small). We're now ready to perform screening!


```python
lam = 0.0

learning_rate=0.1; n_estimators=10; max_depth=6
gbm = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

var_list, ensemble_list = metric_path.run_paths(num_paths=1, screener=metric_learner, model=gbm, X=xtr,\
                                                class_p=ytr, lam=lam, verbose=True)
chosen_var['metric-screen'] = metric_path.get_selected_vars_as_array(0)
```

    Training lasted for 201 iterations.
    Metric screening coefficients:  [0.48 0.17 0.05 0.05 0.04 0.  ]
    New variables:  [  5 189   0   6  13 195]
    Total of 6 variables selected so far. Variables selected in round 0:  [  5 189   0   6  13 195]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 326 iterations.
    Metric screening coefficients:  [0.53 0.16 0.12 0.01]
    New variables:  []
    Total of 6 variables selected so far. Variables selected in round 1:  [  5 189   6  13]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 326 iterations.
    Metric screening coefficients:  [0.51 0.18 0.05 0.04 0.01]
    New variables:  []
    Total of 6 variables selected so far. Variables selected in round 2:  [  5 189   0   6  13]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 201 iterations.
    Metric screening coefficients:  [0.5  0.17 0.05 0.04 0.02]
    New variables:  []
    Total of 6 variables selected so far. Variables selected in round 3:  [  5 189   6   0  13]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 351 iterations.
    Metric screening coefficients:  [0.48 0.2  0.06 0.06 0.05 0.  ]
    New variables:  [196]
    Total of 7 variables selected so far. Variables selected in round 4:  [  5 189   0   6  13 196]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 301 iterations.
    Metric screening coefficients:  [0.5  0.17 0.04 0.01]
    New variables:  []
    Total of 7 variables selected so far. Variables selected in round 5:  [  5 189   6   0]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 301 iterations.
    Metric screening coefficients:  [0.5  0.15 0.03 0.03 0.02 0.01]
    New variables:  [209]
    Total of 8 variables selected so far. Variables selected in round 6:  [  5 189   6  13   0 209]
    Smallest fitted probability:  0.49
    Largest fitted probability:  0.51
    Training lasted for 276 iterations.
    Metric screening coefficients:  [0.47 0.16 0.1  0.03 0.02 0.01 0.01 0.01]
    New variables:  [194 173]
    Total of 10 variables selected so far. Variables selected in round 7:  [  5 189   6   0 194  13 209 173]
    Variables selected without filtering:  [  5 189   0   6  13 195 196 209 194 173]
    Variables selected after filtering:  [  0 209   5   6 189  13]


When performing $t$-stat screening, we will choose the same number of variables as chosen by metric screening in order to make results comparable.


```python
_, t_pval = scipy.stats.ttest_ind(xtr[ytr==0,:],xtr[ytr==1,:], axis=0)
chosen_var['t-screen'] = np.argsort(t_pval)[:len(chosen_var['metric-screen'])]
```


```python
print("Variables found by metric screening: ", chosen_var['metric-screen'])
print("Variables found by t-stat screening: ", chosen_var['t-screen'])
```

    Variables found by metric screening:  [  0   5   6  13 189 209]
    Variables found by t-stat screening:  [  5 189 195 181  13   1]


We see that metric screening and t-stat screening found 4 common variables (out of 6). But t-stat screening failed to find feature 6. It turns out that feature 6 interacts with feature 5: this interaction is picked up by metric screening but not t-stat screening. In fact the t-statistic for feature 6 only ranks 75th out of 214 features.


```python
np.where(np.argsort(t_pval)==6)[0]
```




    array([75])



Finally, let's evaluate the quality of the variables we've found by using them to fit various models.


```python
model_names = ['logistic', 'gbm', 'rf', 'nnet', 'svm']

model_id = dict()
for k, mod in enumerate(model_names): model_id[mod] = k

models = dict()
models['logistic'] = sklearn.linear_model.LogisticRegression()
models['gbm'] = sklearn.ensemble.GradientBoostingClassifier()
models['rf'] = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_features='sqrt')
models['nnet'] = MLPClassifier(max_iter=2000)
models['svm'] = SVC()
```


```python
class_err = np.ones((len(model_names), len(screening_methods)))

def assess_model_with_screening(xtr, ytr, xtes, ytes, model, chosen_var, default_err):
    if len(chosen_var) == 0:
        return default_err
    model.fit(X=xtr[:, chosen_var], y=ytr)
    return np.mean(ytes != model.predict(X=xtes[:, chosen_var]))
```


```python
for screen in screening_methods:
    for mod in model_names:
        class_err[model_id[mod], screener_id[screen]] = assess_model_with_screening(xtr, ytr, xtes, ytes,\
                                                                models[mod], chosen_var[screen], np.mean(ytes))
```


```python
cols = ['no screen', 'tstat-screen', 'metric-screen', 'oracle']
rows = ['lasso', 'boosting', 'rf', 'nnet', 'svm']
pd.DataFrame(np.round(class_err*100, decimals=2), index=rows, columns=cols)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no screen</th>
      <th>tstat-screen</th>
      <th>metric-screen</th>
      <th>oracle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lasso</th>
      <td>48.27</td>
      <td>39.53</td>
      <td>40.00</td>
      <td>36.13</td>
    </tr>
    <tr>
      <th>boosting</th>
      <td>33.33</td>
      <td>34.80</td>
      <td>31.20</td>
      <td>25.20</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>38.07</td>
      <td>35.07</td>
      <td>29.73</td>
      <td>22.13</td>
    </tr>
    <tr>
      <th>nnet</th>
      <td>42.33</td>
      <td>32.60</td>
      <td>29.93</td>
      <td>17.73</td>
    </tr>
    <tr>
      <th>svm</th>
      <td>41.53</td>
      <td>34.80</td>
      <td>30.87</td>
      <td>23.73</td>
    </tr>
  </tbody>
</table>
</div>



We see that metric screening outperforms a linear screening algorithm like t-stat screening especially in terms of being able to train neural networks and SVMs in the presence of noise features.
