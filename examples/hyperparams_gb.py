import os
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

import catboost
from catboost import CatBoostClassifier
from catboost.datasets import amazon

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from tg_bot_ml.table_bot import TGTableSummaryWriter

def cross_val_score(clf, X, y, cat_features, cv=5):
    kfold = StratifiedKFold(cv, shuffle=True, random_state=42)

    _log_losses = []
    _auc_scores = []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        cb = deepcopy(clf)

        cb.fit(X_train, y_train, cat_features=cat_features)
        y_pred = cb.predict_proba(X_test)[:, 1]

        _log_loss = log_loss(y_test, y_pred)
        _roc_auc_score = roc_auc_score(y_test, y_pred)

        _log_losses.append(_log_loss)
        _auc_scores.append(_roc_auc_score)

    return _log_losses, _auc_scores

def objective(params):
    model = CatBoostClassifier(**params, random_state=42, verbose=0)

    log_losses, auc_scores = cross_val_score(model, X, y, cat_features=cat_values, cv=3)

    params['LogLoss'] = np.mean(log_losses)
    params['AucScore'] = np.mean(auc_scores)

    writer.add_record(**params)
    writer.send(sort_by='AucScore', ascending=False)

    return -np.mean(auc_scores)


if __name__ == '__main__':
    X = pd.read_csv('./orange_small_churn_train_data.csv', index_col='ID')
    X = X[~X.labels.isna()]
    X, y = X.drop('labels', axis=1), X.labels

    cat_values = X.select_dtypes('object').columns.tolist()
    X[cat_values] = X[cat_values].fillna('NAN')
    
    writer = TGTableSummaryWriter('../credentials.yaml', 'Catboost Hyperopt : Customer Churn')
    
    search_space = {'learning_rate': hp.uniform('learning_rate', 1e-3, 1e-1),
                'iterations': hp.randint('iterations',100,700),
                'l2_leaf_reg': hp.randint('l2_leaf_reg',0,10),
                'depth': hp.randint('depth',2, 7),
                'bootstrap_type' : hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli'])}


    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=35)
    
    hyperparams = space_eval(search_space, best_params)
    print(hyperparams)