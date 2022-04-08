from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pandas as pd
from preprocess import preprocessor as p
import constants
from text import bow_tfidf
from utils import stratify_data

loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty = ['l1', 'l2', 'elasticnet']
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
class_weight = [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6}, {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}, 'balanced']
eta0 = [1, 10, 100]

param_distributions_sgd = dict(
    loss=loss,
    penalty=penalty,
    alpha=alpha,
    learning_rate=learning_rate,
    class_weight=class_weight,
    eta0=eta0)

n_estimators = [10, 20, 30, 50, 100, 200]
criterion = ["gini", "entropy"]
min_samples_leaf = [1, 3, 10]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}, {'balanced'}]

param_distributions_et = dict(
    n_estimators=n_estimators,
    criterion=criterion,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight
)


def randomized_search_sgd(data, label, param_distributions_sgd):
    sgd = SGDClassifier(max_iter=10000)
    random = RandomizedSearchCV(
        estimator=sgd,
        param_distributions=param_distributions_sgd,
        scoring='accuracy',
        verbose=1,
        n_iter=1000)
    random_result = random.fit(data, label)
    print('Best Score: ', random_result.best_score_)
    print('Best Params: ', random_result.best_params_)


def grid_search_sgd(data, label, param_distributions_sgd):
    sgd = SGDClassifier(max_iter=10000)
    grid = GridSearchCV(
        estimator=sgd,
        param_grid=param_distributions_sgd,
        scoring='roc_auc',
        verbose=1)
    grid_result = grid.fit(data, label)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)


def random_search_et(data, label, param_distributions_et):
    etc = ExtraTreesClassifier()
    random = RandomizedSearchCV(
        estimator=etc,
        param_distributions=param_distributions_et,
        scoring='accuracy',
        verbose=1,
        n_iter=10000)
    random_result = random.fit(data, label)
    print('Best Score: ', random_result.best_score_)
    print('Best Params: ', random_result.best_params_)


def grid_search_et(data, label, param_distributions_et):
    etc = ExtraTreesClassifier()
    grid = GridSearchCV(
        estimator=etc,
        param_grid=param_distributions_et,
        scoring='accuracy',
        verbose=1)
    grid_result = grid.fit(data, label)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)


if __name__ == '__main__':
    train_id = 'columbia'
    experiment = 'email_domains'
    feats_appendix = f'_features_{experiment}'

    data = pd.read_csv(f'../data/{train_id}_enron{feats_appendix}.csv', sep='\t', encoding='utf8')
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    print(data.head())

    test_data = data

    print(data[constants.LABEL].value_counts())
    data = p.prepare_email_content(data, train_id, normalize=True)
    data = p.clean_data(data, 'content')
    print(data[constants.LABEL].value_counts())

    idx_train, idx_test = stratify_data(data.index)

    train_X, train_y, train_meta = data.loc[idx_train, experiment], data.loc[idx_train, constants.LABEL], \
                                   data.loc[idx_train, constants.META_FEATURES]
    test_X, test_y, test_meta = test_data.loc[idx_test, experiment], test_data.loc[idx_test, constants.LABEL], \
                                test_data.loc[idx_test, constants.META_FEATURES]

    dict_vectors, train_processed_docs, test_processed_docs = bow_tfidf.create_bow(train_X, test_X, experiment,
                                                                                   lemmatize=True,
                                                                                   stop_words=False)

    dict_tfidf_vectors = bow_tfidf.create_tfidfs(train_processed_docs, test_processed_docs, experiment)
    dict_vectors.update(dict_tfidf_vectors)

    train_X = dict_vectors['TRAIN_TFIDF']
    randomized_search_sgd(train_X, train_y, param_distributions_sgd)
    grid_search_sgd(train_X, train_y, param_distributions_sgd)
    random_search_et(train_X, train_y, param_distributions_et)
    grid_search_et(train_X, train_y, param_distributions_et)
