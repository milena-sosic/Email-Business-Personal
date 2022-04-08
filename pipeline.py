import argparse
from exploration.visualization import plot_features_distribution, plot_sequence_length
from exploration.scattertext_plots import *
from modeling.standard_models import stochastic_gradient_descent, extra_trees
from modeling.network_models import bilstm_attention, bilstm
from preprocess.feature_selector import select_meta_features
from utils import stratify_data, get_stripped_string
from preprocess.feature_extractor import *
from preprocess import preprocessor as p, normalization
from text import bow_tfidf
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from constants import *
import logging
from datetime import datetime
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


def extract_exp_type(experiment):
    try:
        exp_option = exp_options[experiment]
    except ValueError as e:
        print(
            f'Illegal experiment identifier {args.experiment}. Error: {str(e)}')
        return
    return exp_option


def create_sparse_matrix(meta_features):
    # meta_features = meta_features.replace([np.inf], np.finfo(np.float64).max)
    # meta_features = meta_features.replace([-np.inf], np.finfo(np.float64).min)
    # meta_features.fillna(meta_features.mean(), inplace=True)
    meta_features = csr_matrix(meta_features)
    meta_features = normalization.normalize_l2(meta_features)
    return meta_features


def describe(df, stats):
    d = df.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
    d.loc['IQR'] = d.loc['75%'] - d.loc['25%']
    d.loc['Q1-1.5*IQR'] = d.loc['25%'] - 1.5 * d.loc['IQR']
    d.loc['Q1+1.5*IQR'] = d.loc['75%'] + 1.5 * d.loc['IQR']
    plot_sequence_length(data, experiment, d.loc['75%'][0], d.loc['95%'][0], d.loc['Q1+1.5*IQR'][0])
    return d.append(df.reindex(d.columns, axis=1).agg(stats))


def extract_features(data, data_id, experiment):
    data = data.replace([np.inf, -np.inf, '', ' ', 'nan', np.nan], '')
    print(data[LABEL].value_counts())
    data = p.prepare_email_content(data, data_id, normalize=False)
    print(data[LABEL].value_counts())
    data = p.clean_data(data, experiment)
    print(data[LABEL].value_counts())
    create_lexical_features(data, experiment)
    create_expressions_features(data, experiment)
    create_readability_features(data, experiment)
    create_morality_features(data, experiment)
    create_emotional_features(data, experiment)
    create_ner_count_features(data, experiment)
    plot_features_distribution(data, experiment, data_id)
    data.to_csv(f'data/{data_id}_enron_features_{experiment}.csv', sep='\t', encoding='utf8')
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_id', help='Data for the experiment training: [berkeley, columbia]', type=str,
                        default='columbia')
    parser.add_argument('--test_id', help='Data for the experiment test: [berkeley, columbia]', type=str,
                        default='columbia')
    parser.add_argument('--experiment', help='Name of the Experiment: [E, ED, EQ, EQD, B]', type=str,
                        default='ED')
    parser.add_argument('--preprocess', type=str, help='Flag to preprocess data: [true, false]', default='false')
    parser.add_argument('--visualize', type=str, help='Flag to visualize data: [true, false]', default='false')
    parser.add_argument('--vector_type', help='Name of the Vector Types: [BOW, TFIDF, EMBD, META]', type=str,
                        default='EMBD')
    parser.add_argument('--feature_type',
                        help='Name of the Feature Types: [LEX, CONV, NER, EXP, MOR, EMO, ALL]', type=str,
                        default='ALL')
    parser.add_argument('--lemmatize', type=str, help='Flag to lemmatize content: [true, false]', default='true')
    parser.add_argument('--stop_words', type=str, help='Flag to use stop words: [true, false]', default='true')

    args = parser.parse_args()
    experiment = extract_exp_type(args.experiment)
    vector_type = get_stripped_string(args.vector_type)
    feature_type = get_stripped_string(args.vector_type)
    preprocess = get_stripped_string(args.preprocess) == 'true'
    lemmatize = get_stripped_string(args.lemmatize) == 'true'
    stop_words = get_stripped_string(args.stop_words) == 'true'
    visualize = get_stripped_string(args.visualize) == 'true'

    train_id = get_stripped_string(args.train_id)
    test_id = get_stripped_string(args.test_id)

    date = datetime.now().strftime("%Y%m%d%I%M%S")

    if save_model:
        # will create the folder to save all the models and outputs
        try:
            dir_name = NAME_MODEL_DIR
            os.makedirs(os.path.join(root_dir, dir_name))
            print("The model directory is created")
        except:
            print("The model directory exist and/or can not be created")
        try:
            dir_name = NAME_RESULTS_DIR
            os.makedirs(os.path.join(root_dir, dir_name))
            print("The results directory is created")
        except:
            print("The results directory exists and/or can not be created")

    feats_appendix = '' if preprocess else f'_features_{experiment}'
    data = pd.read_csv(os.path.join(root_dir, NAME_DATA_DIR, f'{train_id}_enron{feats_appendix}.csv'), sep='\t', encoding='utf8')
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    if train_id != test_id:
        test_data = pd.read_csv(os.path.join(root_dir, NAME_DATA_DIR, f'{test_id}_enron{feats_appendix}.csv'), sep='\t', encoding='utf8')
        test_data.drop(test_data.columns[test_data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        test_data = test_data.rename(columns={'to_domains_type': 'recipients_domains_coherency',
                                              'free_domains_splitters_ratio': 'free_domains_ratio'}, inplace=False)
        print(test_data[LABEL].value_counts())
    else:
        test_data = data

    data = data.rename(columns={'to_domains_type': 'recipients_domains_coherency',
                                'free_domains_splitters_ratio': 'free_domains_ratio'}, inplace=False)

    if preprocess:
        data = extract_features(data, train_id, experiment)
        if train_id != test_id:
            extract_features(test_data, test_id, experiment)

    print(data[LABEL].value_counts())
    data = p.prepare_email_content(data, train_id, normalize=True)
    data = p.clean_data(data, experiment)
    print(data[LABEL].value_counts())

    if visualize:
        plot_features_distribution(data, args.experiment, train_id)
        plot_characteristics(data)
        moral_foundations(data)
        plot_phrases(data)
        plot_empath(data)
        plot_word_vectors(data, experiment)
        print(describe(data[['content_lex_count']], ['skew', 'mad', 'kurt']))

    if train_id != test_id:
        print(test_data[LABEL].value_counts())
        test_data = p.prepare_email_content(test_data, test_id, normalize=True)
        test_data = p.clean_data(test_data, experiment)
        print(test_data[LABEL].value_counts())

    if train_id == test_id:
        idx_train, idx_test = stratify_data(data.index, data[LABEL])
        test_data = data
    else:
        idx_train, idx_test = data.index, test_data.index

    train_X, train_y, train_meta = data.loc[idx_train, experiment], data.loc[idx_train, LABEL], \
                                   data.loc[idx_train, META_SELECTION]
    test_X, test_y, test_meta = test_data.loc[idx_test, experiment], test_data.loc[idx_test, LABEL], \
                                test_data.loc[idx_test, META_SELECTION]

    dict_vectors, train_processed_docs, test_processed_docs = bow_tfidf.create_bow(train_X, test_X, experiment,
                                                                                   lemmatize=lemmatize,
                                                                                   stop_words=stop_words)
    if vector_type == 'TFIDF':
        dict_tfidf_vectors = bow_tfidf.create_tfidfs(train_processed_docs, test_processed_docs, experiment)
        dict_vectors.update(dict_tfidf_vectors)

    meta_feats = select_meta_features(train_meta, train_y, args.experiment, train_id, type='drop_col')
    train_meta = train_meta[meta_feats]
    test_meta = test_meta[meta_feats]
    train_meta_columns = train_meta.columns
    dict_vectors['TRAIN_META'] = create_sparse_matrix(train_meta)
    dict_vectors['TEST_META'] = create_sparse_matrix(test_meta)

    dict_meta = dict()
    for key in dict_vectors.keys():
        if ('TRAIN' in key) and ('TRAIN_META' not in key):
            dict_meta[f'{key}_META'] = hstack([dict_vectors['TRAIN_META'], dict_vectors[key]], 'csr')
        elif ('TEST' in key) and ('TEST_META' not in key):
            dict_meta[f'{key}_META'] = hstack([dict_vectors['TEST_META'], dict_vectors[key]], 'csr')
    dict_vectors.update(dict_meta)

    print(dict_vectors.keys())
    for key in dict_vectors.keys():
        print(key, ":", dict_vectors[key].shape)

    results = pd.DataFrame()

    def classify(dict_vectors, train_y, test_y, vector_type, experiment):
        # Traditional models
        print("Traditional models")
        results = stochastic_gradient_descent(dict_vectors[f'TRAIN_{vector_type}'], train_y,
                                              dict_vectors[f'TEST_{vector_type}'], test_y,
                                              f'{vector_type}_{experiment}')
        results.append(extra_trees(dict_vectors[f'TRAIN_{vector_type}'], train_y,
                                   dict_vectors[f'TEST_{vector_type}'], test_y,
                                   f'{vector_type}_{experiment}'))
        return results

    if vector_type == 'BOW':
        # Bow features
        results = results.append(classify(dict_vectors, train_y, test_y, vector_type, args.experiment))
        # Bow + META features
        results = results.append(classify(dict_vectors, train_y, test_y, f"{vector_type}_META", args.experiment))
    elif vector_type == 'TFIDF':
        for key in word_vectors:
            if 'TFIDF' in key:
                # # Tf-idf features
                results = results.append(classify(dict_vectors, train_y, test_y, key, args.experiment))
                # Tf-idf + META features
                results = results.append(classify(dict_vectors, train_y, test_y, f"{key}_META", args.experiment))
    elif vector_type == 'META':
        # META features
        results = results.append(classify(dict_vectors, train_y, test_y, vector_type, args.experiment))
    else:
        """# Word Embeddings - BERT"""
        print("Deep learning models")

        logging.basicConfig(level=logging.INFO)
        print(f'BERT model selected           : {BERT_MODEL}')
        print(f'Preprocess model selected: {PREPROCESS_MODEL}')

        # Neural network models - Bi-LSTM
        results = results.append(
            bilstm(np.array(train_processed_docs), train_y,
                   np.array(test_processed_docs), test_y, experiment=args.experiment, use_meta=False))
        if save_results:
            results.to_csv(os.path.join(root_dir, NAME_RESULTS_DIR,
                                        f'{NAME_RESULTS_FILE}_{vector_type}_{args.experiment}.csv'),
                           sep=";", index=False)

        # Neural network models - Bi-LSTM + META features
        results = results.append(
            bilstm([np.array(train_processed_docs), dict_vectors['TRAIN_META']], train_y,
                   [np.array(test_processed_docs),
                    dict_vectors['TEST_META']], test_y, experiment=args.experiment, use_meta=True))
        if save_results:
            results.to_csv(os.path.join(root_dir, NAME_RESULTS_DIR,
                                        f'{NAME_RESULTS_FILE}_{vector_type}_{args.experiment}.csv'),
                           sep=";", index=False)

        # Neural network models - Bi-LSTM + Attention features
        results = results.append(
            bilstm_attention(np.array(train_processed_docs), train_y,
                             np.array(test_processed_docs), test_y, experiment=args.experiment, use_meta=False))
        if save_results:
            results.to_csv(os.path.join(root_dir, NAME_RESULTS_DIR,
                                        f'{NAME_RESULTS_FILE}_{vector_type}_{args.experiment}.csv'),
                           sep=";", index=False)

        # Neural network models - Bi-LSTM + Attention + META features
        results = results.append(
            bilstm_attention([np.array(train_processed_docs), dict_vectors['TRAIN_META']], train_y,
                             [np.array(test_processed_docs),
                             dict_vectors['TEST_META']], test_y, experiment=args.experiment, use_meta=True))
        if save_results:
            results.to_csv(os.path.join(root_dir, NAME_RESULTS_DIR,
                                        f'{NAME_RESULTS_FILE}_{vector_type}_{args.experiment}_{date}.csv'),
                           sep=";", index=False)

    if save_results:
        results.to_csv(os.path.join(root_dir, NAME_RESULTS_DIR,
                                    f'{NAME_RESULTS_FILE}_{vector_type}_{args.experiment}_{date}.csv'),
                       sep=";", index=False)
