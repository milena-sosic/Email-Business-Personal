from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from constants import TEST_SIZE, SEED
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef, roc_auc_score


def get_stripped_string(x):
    return '' if x is None or x.strip() == '' else x.strip()


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def f1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def class_0_prec(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['0']["precision"]


def class_0_rec(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['0']["recall"]


def class_0_f1(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['0']["f1-score"]


def class_1_prec(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['1']["precision"]


def class_1_rec(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['1']["recall"]


def class_1_f1(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['1']["f1-score"]


score_metrics = {'accuracy': accuracy_score,
                 'balanced_accuracy': balanced_accuracy_score,
                 'prec': precision_score,
                 'recall': recall_score,
                 'f1-score': f1_score,
                 'f1-micro': f1_micro,
                 'f1-macro': f1_macro,
                 'f1-weighted': f1_weighted,
                 'tp': tp,
                 'tn': tn,
                 'fp': fp,
                 'fn': fn,
                 'class_0_prec': class_0_prec,
                 'class_0_rec': class_0_rec,
                 'class_0_f1': class_0_f1,
                 'class_1_prec': class_1_prec,
                 'class_1_rec': class_1_rec,
                 'class_1_f1': class_1_f1,
                 'cohens_kappa': cohen_kappa_score,
                 'matthews_corrcoef': matthews_corrcoef,
                 'roc_auc': roc_auc_score}


def stratify_data(data, label):
    return train_test_split(data, test_size=TEST_SIZE,
                            random_state=SEED, shuffle=True, stratify=label)


def split_data(data):
    return train_test_split(data, test_size=TEST_SIZE,
                            random_state=SEED, shuffle=True)





