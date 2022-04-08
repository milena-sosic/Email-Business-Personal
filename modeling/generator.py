from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import pickle
from official.nlp import optimization
from constants import save_model, VAL_SIZE, TEST_SIZE, INIT_LR
from utils import *
import constants
from exploration.visualization import plot_model_history


def build_model(clf, x, y, X_test, y_test, name='classifier', cv=5, dict_scoring=None, fit_params=None, save=constants.save_model):

    if dict_scoring!=None:
        score = dict_scoring.copy()
        for i in score.keys():
            score[i] = make_scorer(score[i])

    scores = cross_validate(clf, x, y, scoring=score,
                         cv=cv, return_train_score=False,  fit_params=fit_params)

    _model = clf
    _model.fit(x, y)

    if save:
        filename= name+".sav"
        pickle.dump(_model, open(os.path.join(constants.root_dir, constants.NAME_MODEL_DIR, filename), 'wb'))

    index = ["Model", 'vocab_size', 'max_seq_len', 'emb_len', 'epochs', 'batch_size', 'folds', 'val_split', 'test_split']
    value = [name, x.shape[1], 0, 0, 0, 0, cv,
             0 if cv == 1 else int(100/cv), TEST_SIZE]

    for i in scores:  # loop on each metric generate text and values
        if i == "estimator":
            continue
        index.append(i+"_mean")
        value.append(np.mean(scores[i]))

    for i in scores:
        if i == "fit_time":
            continue
        if i == "score_time":
            continue
        scores[i] = np.append(scores[i] ,score[i.split("test_")[-1]](_model, X_test, y_test))
        index.append(i.split("test_")[-1])
        value.append(scores[i][-1])

    return pd.DataFrame(data=value, index=index).T


def build_model_nn(model, X, y, X_test, y_test, name="NN", fit_params=None, scoring=None, n_splits=1,
                    save=save_model, batch_size=16, epochs=10, use_multiprocessing=True, vocab_size=30522,
                   max_seq_len=128, embeddings_len=128, experiment='E'):

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=3)

    if scoring is None:
        dic_score = {}
    else:
        dic_score = scoring.copy()

    scorer = {}
    for i in dic_score.keys():
        scorer[i] = []

    index = ["Model", 'vocab_size', 'max_seq_len', 'emb_len', 'epochs', 'batch_size', 'folds', 'val_split', 'test_split']
    results = [name, vocab_size, max_seq_len, embeddings_len, epochs, batch_size, n_splits,
               VAL_SIZE if n_splits == 1 else int(100/n_splits), TEST_SIZE]


    fit_start = time.time()
    _model = model

    steps_per_epoch = len(y)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=INIT_LR,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    _model.compile(optimizer=optimizer,
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.metrics.BinaryAccuracy())
    _model.fit(X, y, validation_split=VAL_SIZE, epochs=epochs, callbacks=[es], batch_size=batch_size,
               verbose=True, use_multiprocessing=use_multiprocessing)
    if save:
        _model.save_weights(os.path.join(constants.root_dir,
                                         constants.NAME_MODEL_DIR, name + f"_{experiment}"))

    fit_end = time.time() - fit_start

    #generate plots
    plot_name = "_".join([str(x) for x in results[:-3]])
    plot_model_history(_model.history, plot_name)

    score_start = time.time()
    y_pred = (_model.predict(X_test)>0.5).astype(int)
    score_end = time.time() - score_start

    print(f"Precision: {round(100*precision_score(y_test, y_pred, labels=np.unique(y_pred)), 3)}% , "
          f"Recall: {round(100*recall_score(y_test, y_pred), 3)}%, "
          f"Time \t {round(fit_end, 4)} ms")

    index, results = compute_scorer(dic_score, fit_end, index, results, score_end, scorer, y_pred, y_test)

    return pd.DataFrame(results, index=index).T


def compute_scorer(dic_score, fit_end, index, results, score_end, scorer, y_pred, y_test):
    for i in scorer:
        if i == "fit_time":
            continue
        if i == "score_time":
            continue
    for i in dic_score.keys():
        if i == "fit_time":
            scorer[i].append(fit_end)
            index.append(i + '_overall')
            results.append(fit_end)
            continue
        if i == "score_time":
            scorer[i].append(score_end)
            index.append(i + '_overall')
            results.append(score_end)
            continue
        try:
            scorer[i].append(dic_score[i](y_pred, y_test))
        except ValueError:
            pass
        index.append(i)
        if len(scorer[i]) > 0:
            results.append(scorer[i][-1])
        else:
            results.append(0)
    return index, results