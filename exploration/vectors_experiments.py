from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import constants
import string
from preprocess.preprocessor import lemmatize_sentence, lemmatize_sentence_list, restore_contractions
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocessor as p, normalization
from utils import stratify_data
import os
import seaborn as sns
import regex as re


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
    t0 = time()
    # sentiment_fit = pipeline.fit(x_train, y_train)
    # y_pred = sentiment_fit.predict(x_test)
    y_pred = cross_val_predict(pipeline, x_train, y_train, cv=5)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_train, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy * 100))
    print("accuracy score: {0:.2f}%".format(accuracy * 100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-" * 80)
    return accuracy, train_test_time


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stems = lemmatize_sentence_list(text, lemmatizer)
    return stems


def nfeature_accuracy_checker(x_train, y_train, x_test, y_test, vectorizer=CountVectorizer(), tokenizer=None,
                              analyzer='word', n_features=np.arange(5000, 60001, 5000), stop_words=None,
                              ngram_range=(1, 1), min_df=1, max_df=1.0,
                              classifier=SGDClassifier(),
                              label='', color='', linestyle=''):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                              max_features=n, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy, tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_test,
                                                      y_test)
        result.append((n, nfeature_accuracy, tt_time, label, color, linestyle))

    columns = ['nfeatures', 'test_accuracy', 'train_test_time', 'label', 'color', 'linestyle']

    return pd.DataFrame(result, columns=columns)


def arange(length):
    return np.arange(5000, length, length//20)


def compare_vectors(x_train, y_train, x_test, y_test, max_feats):

    tfidf_vec = TfidfVectorizer()
    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 1), label='tfidf word (1,1)',
                                             n_features=n_features, color='orangered', linestyle='solid'))
    n_features = arange(max_feats['tfidf_1_2'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 2), label='tfidf word (1,2)',
                                             n_features=n_features, color='gold', linestyle='solid'))
    n_features = arange(max_feats['tfidf_1_3'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 3), label='tfidf word (1,3)',
                                             n_features=n_features, color='royalblue', linestyle='solid'))
    n_features = arange(max_feats['tfidf_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 4), label='tfidf word (1,4)',
                                             n_features=n_features, color='green', linestyle='solid'))
    n_features = arange(max_feats['count_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 1), label='bow word (1,1)',
                                             n_features=n_features, color='orangered', linestyle=':'))
    n_features = arange(max_feats['count_1_2'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 2), label='bow word (1,2)',
                                             n_features=n_features, color='gold', linestyle=':'))
    n_features = arange(max_feats['count_1_3'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 3), label='bow word (1,3)',
                                             n_features=n_features, color='royalblue', linestyle=':'))
    n_features = arange(max_feats['count_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 4), label='bow word (1,4)',
                                             n_features=n_features, color='green', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")

    plt.title("N-gram(1~4) Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_VEC_TYPE_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')



def compare_stop_words(x_train, y_train, x_test, y_test, max_feats):

    tfidf_vec = TfidfVectorizer()

    personal = []
    with open('../lexicons/personal_names.txt', 'r') as f:
        f_content = f.read()
        # splitting text.txt by non alphanumeric characters
        processed = re.split(', ', f_content)
        # print(processed)
        personal = processed
    print(personal)
    custom = ['enron', 'ect', 'hou', 'com', 'org', 'would', 'could', 'one', 'us', 'say', 'that', 'this',
              'will', 'and', 'have', 'to', 'of',
              '2000', '2001', 'the', 'a', 'is', 'it', 'know', 'let', 'need', 'thank', 'might', 'make', 'want']
    english = stopwords.words('english')

    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, label='tfidf word (1,1)',
                                             n_features=n_features, color='orangered', linestyle='solid'))
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, stop_words=english, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - stop words: english',
                                             n_features=n_features, color='gold', linestyle=':'))

    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, stop_words=custom, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - stop words: custom',
                                             n_features=n_features, color='green', linestyle=':'))

    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, stop_words=custom + english, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - stop words: custom + english',
                                             n_features=n_features, color='royalblue', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o", err_style="bars", ci=68)

    plt.title("Stop Words Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_STOP_WORDS_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


def compare_spec_types(x_train, y_train, x_test, y_test, max_feats):

    tfidf_vec = TfidfVectorizer()
    numbers = []
    punctuations = []

    punct = string.punctuation
    pattern = r"[{}]".format(punct)  # create the pattern
    x_train_n = x_train.apply(lambda x: re.sub(r'([0-9]+)', '', x))
    x_train_p = x_train.apply(lambda x: re.sub(pattern, ' ', x))
    x_train_c = x_train.apply(lambda x: restore_contractions(x))
    #x_train_s = x_train.apply(lambda x: re.sub(r'\s+', ' ', x))

    x_test_n = x_test.apply(lambda x: re.sub(r'([0-9]+)', '', x))
    x_test_p = x_test.apply(lambda x: re.sub(pattern, ' ', x))
    x_test_c = x_test.apply(lambda x: restore_contractions(x))
    #x_test_s = x_test.apply(lambda x: re.sub(r'\s+', ' ', x))

    personal = []
    with open('../lexicons/personal_names.txt', 'r') as f:
        f_content = f.read()
        # splitting text.txt by non alphanumeric characters
        processed = re.split(', ', f_content)
        # print(processed)
        personal = processed


    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, label='tfidf word (1,1)',
                                             n_features=n_features, color='orangered',
                                             linestyle='solid'))
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train_n, y_train, x_test_n, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - norm: numbers',
                                             n_features=n_features, color='gold', linestyle=':'))
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, stop_words=personal, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - norm: personal names',
                                             n_features=n_features, color='purple', linestyle=':'))
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train_p, y_train, x_test_p, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - norm: punctuations',
                                             n_features=n_features, color='green', linestyle=':'))
    results.append(nfeature_accuracy_checker(x_train_c, y_train, x_test_c, y_test,
                                             vectorizer=tfidf_vec, ngram_range=(1, 1),
                                             label='tfidf word (1,1) - norm: contractions',
                                             n_features=n_features, color='royalblue', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")

    plt.title("Special words Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_SPEC_WORD_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


def compare_analyzers(x_train, y_train, x_test, y_test, max_feats):
    tfidf_vec = TfidfVectorizer()

    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, label='tfidf word (1,1)',
                                             n_features=n_features, color='orangered', linestyle='solid'))
    n_features = arange(max_feats['char_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char', ngram_range=(1, 4),
                                             label='tfidf - char (1,4)',
                                             n_features=n_features, color='gold', linestyle=':'))
    n_features = arange(max_feats['char_wb_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char_wb', ngram_range=(1, 4),
                                             label='tfidf - char_wb (1,4)',
                                             n_features=n_features, color='green', linestyle=':'))
    n_features = arange(max_feats['count_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 1),
                                             label='bow word (1,1)',
                                             n_features=n_features, color='royalblue', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")

    plt.title("Tokenizers Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_ANALYZER_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


def compare_min_max(x_train, y_train, x_test, y_test, max_feats):
    tfidf_vec = TfidfVectorizer()

    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, label='tfidf',
                                             n_features=n_features, color='orangered', linestyle='solid'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, min_df=2, max_df=0.95,
                                             label='tfidf - min_df=2, max_df=95%',
                                             n_features=n_features, color='gold', linestyle='solid'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, min_df=2, max_df=0.80,
                                             label='tfidf - min_df=2, max_df=80%',
                                             n_features=n_features, color='green', linestyle='solid'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, min_df=3, max_df=0.80,
                                             label='tfidf - min_df=3, max_df=80%',
                                             n_features=n_features, color='royalblue', linestyle='solid'))
    n_features = arange(max_feats['count_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             label='bow', n_features=n_features, color='orangered',
                                             linestyle=':'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             min_df=2, max_df=0.95,
                                             label='bow - min_df=2, max_df=95%',
                                             n_features=n_features, color='gold', linestyle=':'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             min_df=2, max_df=0.80,
                                             label='bow - min_df=2, max_df=80%',
                                             n_features=n_features, color='green', linestyle=':'))

    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             min_df=3, max_df=0.80,
                                             label='bow - min_df=3, max_df=80%',
                                             n_features=n_features, color='royalblue', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")
    plt.title("MIN_DF/MAX_DF Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_MIN_MAX_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


def compare_lemmatization(x_train, y_train, x_test, y_test, max_feats):

    tfidf_vec = TfidfVectorizer()

    results = []
    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, label='tfidf (1,1)',
                                             n_features=n_features, color='orangered',
                                             linestyle='solid'))
    n_features = arange(max_feats['char_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char', ngram_range=(1, 4),
                                             label='tfidf - char (1,4)',
                                             n_features=n_features, color='gold', linestyle='solid'))

    n_features = arange(max_feats['count_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             ngram_range=(1, 1),
                                             label='bow (1,1)',
                                             n_features=n_features, color='royalblue', linestyle='solid'))

    n_features = arange(max_feats['tfidf_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, tokenizer=tokenize, label='tfidf (1,1) - lemma',
                                             n_features=n_features, color='orangered',
                                             linestyle=':'))
    n_features = arange(max_feats['char_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, tokenizer=tokenize, analyzer='char', ngram_range=(1, 4),
                                             label='tfidf - char (1,4) - lemma',
                                             n_features=n_features, color='gold', linestyle=':'))

    n_features = arange(max_feats['count_1_1'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test, tokenizer=tokenize,
                                             ngram_range=(1, 1),
                                             label='bow (1,1) - lemma',
                                             n_features=n_features, color='royalblue', linestyle=':'))

    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")

    plt.title("Lemmatization Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_LEMMA_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


def compare_chars(x_train, y_train, x_test, y_test, max_feats):
    tfidf_vec = TfidfVectorizer()

    results = []

    n_features = arange(max_feats['char_1_3'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char', ngram_range=(1, 3),
                                             label='tfidf - char (1,3)',
                                             n_features=n_features, color='royalblue', linestyle='solid'))
    n_features = arange(max_feats['char_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char', ngram_range=(1, 4),
                                             label='tfidf - char (1,4)',
                                             n_features=n_features, color='orangered', linestyle='solid'))
    n_features = arange(max_feats['char_1_5'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char', ngram_range=(1, 5),
                                             label='tfidf - char (1,5)',
                                             n_features=n_features, color='gold', linestyle='solid'))

    n_features = arange(max_feats['char_wb_1_3'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char_wb', ngram_range=(1, 3),
                                             label='tfidf - char_wb (1,3)', n_features=n_features, color='royalblue',
                                             linestyle=':'))
    n_features = arange(max_feats['char_wb_1_4'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char_wb', ngram_range=(1, 4),
                                             label='tfidf - char_wb (1,4)',
                                             n_features=n_features, color='orangered', linestyle=':'))
    n_features = arange(max_feats['char_wb_1_5'])
    results.append(nfeature_accuracy_checker(x_train, y_train, x_test, y_test,
                                             vectorizer=tfidf_vec, analyzer='char_wb', ngram_range=(1, 5),
                                             label='tfidf - char_wb (1,5)',
                                             n_features=n_features, color='gold', linestyle=':'))


    plt.figure(figsize=(8, 6))
    for df in results:
        lineplot = sns.lineplot(x='nfeatures', y="test_accuracy", data=df, label=df.label.any(),
                                color=df.color.any(), linestyle=df.linestyle.any(), marker="o")

    plt.title("Char Analyzers Experiment")
    plt.xlabel("Number of features")
    plt.ylabel("5-Fold Cross Validation Accuracy")
    plt.legend(frameon=False, loc='lower right')
    sns.despine()
    file_path = os.path.join(constants.root_dir, constants.NAME_RESULTS_DIR, constants.NAME_PLOTS_DIR,
                             constants.NAME_FEATURE_DIR, f'EXP_CHR_ANALYZER_{experiment}')
    lineplot.figure.savefig(file_path, bbox_inches='tight')


if __name__ == '__main__':
    train_id = 'columbia'
    experiment = 'email_domains'
    meta_feats_list = []

    feats_appendix = f'_features_{experiment}'

    data = pd.read_csv(f'../data/{train_id}_enron{feats_appendix}.csv', sep='\t', encoding='utf8')
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    data = data.rename(columns={'to_domains_type': 'recipients_domains_coherency',
                                'free_domains_splitters_ratio': 'free_domains_ratio'}, inplace=False)


    test_data = data

    data = data.replace([np.inf, -np.inf, '', ' ', 'nan', np.nan], '')
    data = p.prepare_email_content(data, train_id, normalize=False)
    data = p.clean_data(data, experiment)
    print(data[constants.LABEL].value_counts())

    idx_train, idx_test = stratify_data(data.index, data[constants.LABEL])


    train_X, train_y, train_meta = data.loc[idx_train, experiment], data.loc[idx_train, constants.LABEL], \
                                   data.loc[idx_train, constants.META_FEATURES]
    test_X, test_y, test_meta = test_data.loc[idx_test, experiment], test_data.loc[idx_test, constants.LABEL], \
                                test_data.loc[idx_test, constants.META_FEATURES]


    max_feats = dict()
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(train_X)
    maxf_count_1_1 = len(count_vect.vocabulary_)
    max_feats['count_1_1'] = maxf_count_1_1 if maxf_count_1_1 <= 100000 else 100000

    count_vect = CountVectorizer(analyzer='word', ngram_range=(1,2))
    count_vect.fit(train_X)
    maxf_count_1_2 = len(count_vect.vocabulary_)
    max_feats['count_1_2'] = maxf_count_1_2 if maxf_count_1_2 <= 100000 else 100000

    count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    count_vect.fit(train_X)
    maxf_count_1_3 = len(count_vect.vocabulary_)
    max_feats['count_1_3'] = maxf_count_1_3 if maxf_count_1_3 <= 100000 else 100000

    count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 4))
    count_vect.fit(train_X)
    maxf_count_1_4 = len(count_vect.vocabulary_)
    max_feats['count_1_4'] = maxf_count_1_4 if maxf_count_1_4 <= 100000 else 100000

    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,1))
    tfidf_vect.fit(train_X)
    maxf_tfidf_1_1 = len(tfidf_vect.vocabulary_)
    max_feats['tfidf_1_1'] = maxf_tfidf_1_1 if maxf_tfidf_1_1 <= 100000 else 100000

    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    tfidf_vect.fit(train_X)
    maxf_tfidf_1_2 = len(tfidf_vect.vocabulary_)
    max_feats['tfidf_1_2'] = maxf_tfidf_1_2 if maxf_tfidf_1_2 <= 100000 else 100000

    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
    tfidf_vect.fit(train_X)
    maxf_tfidf_1_3 = len(tfidf_vect.vocabulary_)
    max_feats['tfidf_1_3'] = maxf_tfidf_1_3 if maxf_tfidf_1_3 <= 100000 else 100000

    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 4))
    tfidf_vect.fit(train_X)
    maxf_tfidf_1_4 = len(tfidf_vect.vocabulary_)
    max_feats['tfidf_1_4'] = maxf_tfidf_1_4 if maxf_tfidf_1_4 <= 100000 else 100000

    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    tfidf_vect_ngram_chars.fit(train_X)
    maxf_char_1_2 = len(tfidf_vect_ngram_chars.vocabulary_)
    max_feats['char_1_2'] = maxf_char_1_2 if maxf_char_1_2 <= 100000 else 100000

    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    tfidf_vect_ngram_chars.fit(train_X)
    maxf_char_1_3 = len(tfidf_vect_ngram_chars.vocabulary_)
    max_feats['char_1_3'] = maxf_char_1_3 if maxf_char_1_3 <= 100000 else 100000

    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
    tfidf_vect_ngram_chars.fit(train_X)
    maxf_char_1_4 = len(tfidf_vect_ngram_chars.vocabulary_)
    max_feats['char_1_4'] = maxf_char_1_4 if maxf_char_1_4 <= 100000 else 100000

    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))
    tfidf_vect_ngram_chars.fit(train_X)
    maxf_char_1_5 = len(tfidf_vect_ngram_chars.vocabulary_)
    max_feats['char_1_5'] = maxf_char_1_5 if maxf_char_1_5 <= 100000 else 100000

    tfidf_vect_ngram_chars_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2))
    tfidf_vect_ngram_chars_wb.fit(train_X)
    maxf_char_wb_1_2 = len(tfidf_vect_ngram_chars_wb.vocabulary_)
    max_feats['char_wb_1_2'] = maxf_char_wb_1_2 if maxf_char_wb_1_2 <= 100000 else 100000

    tfidf_vect_ngram_chars_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
    tfidf_vect_ngram_chars_wb.fit(train_X)
    maxf_char_wb_1_3 = len(tfidf_vect_ngram_chars_wb.vocabulary_)
    max_feats['char_wb_1_3'] = maxf_char_wb_1_3 if maxf_char_wb_1_3 <= 100000 else 100000

    tfidf_vect_ngram_chars_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4))
    tfidf_vect_ngram_chars_wb.fit(train_X)
    maxf_char_wb_1_4 = len(tfidf_vect_ngram_chars_wb.vocabulary_)
    max_feats['char_wb_1_4'] = maxf_char_wb_1_4 if maxf_char_wb_1_4 <= 100000 else 100000

    tfidf_vect_ngram_chars_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5))
    tfidf_vect_ngram_chars_wb.fit(train_X)
    maxf_char_wb_1_5 = len(tfidf_vect_ngram_chars_wb.vocabulary_)
    max_feats['char_wb_1_5'] = maxf_char_wb_1_5 if maxf_char_wb_1_5 <= 100000 else 100000

    for key in max_feats.keys():
        max_feats[key] += 5000
    compare_vectors(train_X, train_y, test_X, test_y, max_feats)
    compare_stop_words(train_X, train_y, test_X, test_y, max_feats)
    compare_analyzers(train_X, train_y, test_X, test_y, max_feats)
    compare_min_max(train_X, train_y, test_X, test_y, max_feats)
    compare_spec_types(train_X, train_y, test_X, test_y, max_feats)
    compare_lemmatization(train_X, train_y, test_X, test_y, max_feats)
    compare_chars(train_X, train_y, test_X, test_y, max_feats)