from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from constants import *
from modeling.generator import build_model
from utils import *


def stochastic_gradient_descent(train_data, train_y, test_data, test_y, type):
    sgd = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001,
                        class_weight={1:0.7, 0:0.3}, eta0=10, learning_rate='optimal', n_jobs=-1)
    print(f"SGD_{type}")
    result = build_model(sgd, train_data, train_y, test_data, test_y, name=f"SGD_{type}",
                         cv=FOLDS, dict_scoring=score_metrics, save=save_model)
    return result


def extra_trees(train_data, train_y, test_data, test_y, type):
    et = ExtraTreesClassifier(n_estimators=10, min_samples_leaf=3, criterion='entropy', class_weight={1:0.7, 0:0.3})
    print(f"ET_{type}")
    result = build_model(et, train_data, train_y, test_data, test_y, name=f"ET_{type}",
                         cv=FOLDS, dict_scoring=score_metrics, save=save_model)
    return result
