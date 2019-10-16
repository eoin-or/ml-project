import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from bayes_opt import BayesianOptimization

# Reading in training data
train_df = pd.read_csv('training.csv')

# Dropping the 'instance' feature, since it will obviously have no effect
X_train = train_df.drop('Instance', axis=1)
X_train = X_train.drop('Income in EUR', axis=1)
y_train = train_df['Income in EUR']

X_test = pd.read_csv('test.csv')
X_test = X_test.drop('Income', axis=1)
X_test = X_test.drop('Instance', axis=1)


# This imputes missing values, using the median value for numerical columns and simply the most frequent value for categorical columns.
ct = ColumnTransformer(transformers=[('num_imp', SimpleImputer(strategy='median'), [0, 2, 4, 9]), ('cat_imp', SimpleImputer(strategy='most_frequent'), [1, 3, 5, 6, 7, 8])], remainder='passthrough')
ct.fit(X_train, y_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)


# Adding in a new column which signifies whether someone has a 'senior' job, since those are more likely to be high-paying.
# This is based on having one of a certain set of terms in their job title.
jobs = X_train[:,6]
senior_job_terms = ['senior', 'manager', 'doctor', 'lawyer', 'analyst', 'programmer', 'specialist', 'supervisor', 'chief']
senior_job = []
for j in jobs:
    found = False
    for s in senior_job_terms:
        if s in j:
            senior_job.append(True)
            found = True
            break
    if not found:
        senior_job.append(False)
                                                    
X_train = np.column_stack((X_train, senior_job))

# Encoding categorical variables as floats using target encoding, essentially the probability of each value of said variable
enc = ce.TargetEncoder(cols=[4, 5, 6, 7, 8, 9, 10]).fit(X_train, y_train)
X_train = enc.transform(X_train)

# XGBoost works fastest with its own DMatrix data structure.
dtrain = xgb.DMatrix(X_train, label=y_train)

# Hyperparameter tuning.
# The function we want to optimise with Bayesian optimisation. Since it will try to maximise rather than minimise the value, we have to return the *negative* RMSE score
def xgb_evaluate(max_depth, gamma, colsample_bytree, subsample):
    params = {'eval_metric': 'rmse',
            'max_depth': int(max_depth),
            'eta': 0.1,
            'subsample': subsample,
            'gamma': gamma,
            'colsample_bytree': colsample_bytree}

    cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5)
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

# Specifying the bounds of each hyperparameter we want to tune. 
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
    'gamma': (0, 1),
    'colsample_bytree': (0.3, 0.9),
    'subsample': (0.5, 0.9)})

xgb_bo.maximize(init_points=10, n_iter=20, acq='ei')

# Specifiying we want to use the best params which were found during the Bayesian optimisation step.
params = xgb_bo.max['params']
params['max_depth'] = int(params['max_depth'])

model = xgb.train(params, dtrain, num_boost_round=250)

# Adding the senior_jobs column to the test data
jobs = X_test[:,6] 
senior_job = []
for j in jobs:
    found=False
    for s in senior_job_terms:
        if s in j:
            senior_job.append(True)
            found = True
            break
    if not found:
        senior_job.append(False)

X_test = np.column_stack((X_test, senior_job))
x_test = enc.transform(X_test)

X_test = xgb.DMatrix(X_test)
predicted_scores = model.predict(X_test)

# Writing to the output file
with open('tcd ml 2019-20 income prediction submission file.csv', 'wb') as f:
        index = 111994
        f.write(b'Instance,Income\n')
        for p in predicted_scores:
            f.write(str(index).encode())
            f.write(b',')
            f.write(str(p).encode())
            f.write(b'\n')
            index += 1
