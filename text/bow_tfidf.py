from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import constants
from constants import NAME_TF_IDF_MODEL, NAME_TF_IDF_NGRAM_MODEL, NAME_TF_IDF_NGRAM_CHAR_MODEL, NAME_COUNT_VECT_MODEL, \
    root_dir, save_model
from preprocess.preprocessor import lemmatize_sentence
from nltk.stem import WordNetLemmatizer
import os


def create_bow(train, test, experiment, lemmatize=True, stop_words=False):
    lemmatizer = WordNetLemmatizer()
    if stop_words:
        stop_words = constants.STOP_WORDS + stopwords.words('english')
    else:
        stop_words = set()

    raw_docs_train = train.astype(str).tolist()
    raw_docs_test = test.astype(str).tolist()

    print(stop_words)
    processed_docs_train = []
    for doc in raw_docs_train:
        if lemmatize:
            doc = lemmatize_sentence(doc, lemmatizer)
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_train.append(" ".join(filtered))

    processed_docs_test = []
    for doc in raw_docs_test:
        if lemmatize:
            doc = lemmatize_sentence(doc, lemmatizer)
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_test.append(" ".join(filtered))

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')  # min_df=2, max_df=0.8,
    count_vect.fit(processed_docs_train)
    print("bow done")

    train_count = count_vect.transform(processed_docs_train)
    test_count = count_vect.transform(processed_docs_test)

    if save_model:
        # save the model to disk
        filename = NAME_COUNT_VECT_MODEL + f'_{experiment}.sav'
        pickle.dump(count_vect, open(os.path.join(constants.root_dir,
                                                  constants.NAME_MODEL_DIR, 'vectorizers', filename), 'wb'))
    dict_vectors = dict()
    dict_vectors['TRAIN_BOW'] = train_count
    dict_vectors['TEST_BOW'] = test_count
    return dict_vectors, processed_docs_train, processed_docs_test


def create_tfidfs(train, test, experiment):
    dict_vectors = dict()
    # # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tfidf_vect.fit(train)
    train_tfidf = tfidf_vect.transform(train)
    test_tfidf = tfidf_vect.transform(test)
    dict_vectors['TRAIN_TFIDF'] = train_tfidf
    dict_vectors['TEST_TFIDF'] = test_tfidf
    print("word level tf-idf done")

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))  # token_pattern=r'\w{1,}',
    tfidf_vect_ngram.fit(train)
    train_tfidf_ngram = tfidf_vect_ngram.transform(train)
    test_tfidf_ngram = tfidf_vect_ngram.transform(test)
    dict_vectors['TRAIN_TFIDF_NGRAM'] = train_tfidf_ngram
    dict_vectors['TEST_TFIDF_NGRAM'] = test_tfidf_ngram
    print("ngram level tf-idf done")

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))  # token_pattern=r'\w{1,}',
    tfidf_vect_ngram_chars.fit(train)
    train_tfidf_ngram_char = tfidf_vect_ngram_chars.transform(train)
    test_tfidf_ngram_char = tfidf_vect_ngram_chars.transform(test)
    dict_vectors['TRAIN_TFIDF_NGRAM_CHAR'] = train_tfidf_ngram_char
    dict_vectors['TEST_TFIDF_NGRAM_CHAR'] = test_tfidf_ngram_char
    print("characters level tf-idf done")

    if save_model:
        # save the tf-idf vectorizer to disk
        filename = NAME_TF_IDF_MODEL + f'_{experiment}.sav'
        pickle.dump(tfidf_vect, open(os.path.join(root_dir, constants.NAME_MODEL_DIR, 'vectorizer', filename), 'wb'))

        # save the model tf-idf ngram vectorizer to disk
        filename = NAME_TF_IDF_NGRAM_MODEL + f'_{experiment}.sav'
        pickle.dump(tfidf_vect_ngram,
                    open(os.path.join(root_dir, constants.NAME_MODEL_DIR, 'vectorizer', filename), 'wb'))

        # save the model tf-idf ngram char vectorizer to disk
        filename = NAME_TF_IDF_NGRAM_CHAR_MODEL + f'_{experiment}.sav'
        pickle.dump(tfidf_vect_ngram_chars,
                    open(os.path.join(root_dir, constants.NAME_MODEL_DIR, 'vectorizer', filename), 'wb'))
    return dict_vectors
