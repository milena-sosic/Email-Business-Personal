import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from exploration.visualization import plot_features_importance, plot_features_corr
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import numpy as np
from datetime import datetime


def drop_col_feat_imp(model, X_train, y_train, random_state=42):
    model_clone = clone(model)
    model_clone.random_state = random_state
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    importances = []

    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(drop_col_score - benchmark_score)

    importances_df = imp_df(X_train.columns, importances)

    return importances_df


def imp_df(features, importances):
    return pd.concat(
        (pd.DataFrame(features, columns=['Feature']),
         pd.DataFrame(importances, columns=['Importance'])), axis=1).sort_values(by='Importance', ascending=False)


def select_meta_features(X, y, experiment, train_id, type='drop_col'):
    date = datetime.now().strftime("%Y%m%d%I%M%S")
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    plot_features_corr(X, hierarchy, corr, corr_linkage, experiment, train_id, date)
    cluster_ids = hierarchy.fcluster(corr_linkage, 0.7, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    X = X.iloc[:, selected_features]

    model = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001,
                          class_weight={1: 0.7, 0: 0.3}, eta0=10, learning_rate='optimal', n_jobs=-1)

    if type == 'drop_col':
        full_list = drop_col_feat_imp(model, X, y, random_state=42)
        perm_df = None
        plot_type = 'DROP_COL'
    else:
        full_list, perm_df = perm_feat_imp(model, X, y)
        plot_type = 'PERM'
    plot_features_importance(full_list, perm_df, plot_type, experiment, train_id, date)

    print(len(X.columns))
    full_list = full_list[np.abs(full_list.Importance) > 0.01]
    print(len(full_list.Feature.tolist()))
    print(full_list.Feature.tolist())
    print(full_list)
    return full_list.Feature.tolist()


def perm_feat_imp(model, X, y):
    model.fit(X, y)
    results = permutation_importance(model, X, y, n_repeats=10, random_state=42)

    perm_sorted_idx = results.importances_mean.argsort()[::-1]

    perm_train_df = pd.DataFrame(data=results.importances[perm_sorted_idx].T,
                                 columns=X.columns[perm_sorted_idx])
    full_list = imp_df(X.columns[perm_sorted_idx], results.importances_mean[perm_sorted_idx])
    return full_list, perm_train_df
