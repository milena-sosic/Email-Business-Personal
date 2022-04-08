TEXT = "content"
LABEL = "email_final_label"
NAME_DATA_DIR = "data"
NAME_MODEL_DIR = "models"
NAME_RESULTS_DIR = "results"
NAME_PLOTS_DIR = "plots"
NAME_FEATURE_DIR = "features"
NAME_RESULTS_FILE = "models_results"

BATCH_SIZE = 64
SEED = 42
NUM_FILTERS = 64
EPOCHS = 30
INIT_LR = 3e-5 #1e-3, 2e-5
MAX_SEQ_LEN = 256

# Place here the path to the models
BERT_MODEL = './models/bert/small_bert_bert_en_uncased_L-2_H-256_A-4_2'     # Path to the model saved on local disk
#'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2'  # Path to the model on TensorFlow Hub

PREPROCESS_MODEL = './models/bert/bert_en_uncased_preprocess_3'     # Path to the model saved on local disk
#'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'        # Path to the model on TensorFlow Hub

BERT_MODEL_LIST = [
'../models/bert/small_bert_bert_en_uncased_L-2_H-256_A-4_2',
'../models/bert/small_bert_bert_en_uncased_L-2_H-768_A-12_2',
'../models/bert/small_bert_bert_en_uncased_L-4_H-512_A-8_2',
'../models/bert/small_bert_bert_en_uncased_L-8_H-256_A-4_2',
'../models/bert/small_bert_bert_en_uncased_L-10_H-128_A-2_2'
]

MAX_SEQ_LEN_LIST = [128, 256, 512]
DEFAULT_MAIL_DOMAIN_ENRON = 'enron.com'
STOP_WORDS = ['enron', 'ect', 'hou', 'com', 'org', 'would', 'could', 'one', 'us', 'say', 'that', 'this', 'will', 'and', 'have', 'to', 'of',
              '2000', '2001', 'the', 'a', 'is', 'it', 'know', 'let', 'need', 'thank', 'might', 'make', 'want']

FOLDS = 5        # Number of splits for cross-validation and k-folds
TEST_SIZE = 0.25    # Test set size
VAL_SIZE = 0.25     # Validation set size
save_results = True     # Flag to save an output file containing the results
save_model = False     # Flag to save a model
root_dir = "../Emails_Business_Personal"    # Path to the project location

# Name files
NAME_BERT_EMBEDDINGS = "BERT_embeddings"
NAME_COUNT_VECT_MODEL = "BOW_model"
NAME_TF_IDF_MODEL = "TF_IDF_model"
NAME_TF_IDF_NGRAM_MODEL = "TF_IDF_NGRAM_model"
NAME_TF_IDF_NGRAM_CHAR_MODEL = "TF_IDF_NGRAM_CHAR_model"
NAME_BERT_TOKENS = "BERT_tokens"


SYNT_FEATURES = ['content_lex_count', 'subject_lex_count',
                'content_lex_length', 'subject_lex_length',
                'sentences', 'avg_word_length', 'avg_sentence_length',
                'noun_phrases_ratio', 'difficult_words_ratio',
                'words_density','sentences_density', 'newlines',
                'syllable_count', 'avg_syll_per_word','avg_syll_per_sentence',
                'acronyms_indicator', 'business_indicator']

NER_FEATURES = ['names_lex_ratio', 'orgs_lex_ratio',
                'numbers_lex_ratio', 'conns_lex_ratio',
                'links_lex_ratio', 'emails_lex_ratio']

PUNCT_FEATURES = ['question_marks_ratio', 'exclamation_marks_ratio', 'dots_ratio',
                  'hash_tags_ratio', 'ref_tags_ratio']

LEX_FEATURES = SYNT_FEATURES + NER_FEATURES + PUNCT_FEATURES

CONV_FEATURES = ['free_domains_ratio',
                 'recipients_count',
                 'recipients_domains_coherency']

READ_FEATURES = ['flesch_reading_ease',
                 'automated_readability_index',
                 'linsear_write_formula']

SENTI_FEATURES = ['polarity', 'subjectivity']

EXPRESS_FEATURES  = READ_FEATURES + SENTI_FEATURES

MORAL_FEATURES = ['care_p', 'fairness_p', 'loyalty_p', 'authority_p', 'sanctity_p',
                  'care_sent', 'fairness_sent', 'loyalty_sent', 'authority_sent',
                  'sanctity_sent', 'moral_nonmoral_ratio']

EMO_FEATURES = ['fear', 'anger', 'trust', 'surprise', 'positive',
                'negative', 'sadness', 'disgust', 'joy']

META_FEATURES = LEX_FEATURES + CONV_FEATURES + NER_FEATURES + \
                EXPRESS_FEATURES + MORAL_FEATURES + EMO_FEATURES


META_SELECTION = ['linsear_write_formula', 'care_sent', 'surprise', 'numbers_lex_ratio', 'automated_readability_index',
                  'question_marks_ratio', 'subject_lex_length', 'polarity', 'recipients_domains_coherency', 'conns_lex_ratio',
                  'dots_ratio', 'avg_word_length', 'positive', 'recipients_count', 'moral_nonmoral_ratio',
                  'free_domains_ratio', 'subjectivity', 'difficult_words_ratio', 'trust', 'joy', 'business_indicator']

word_vectors = {
    'BOW': 'BOW',
    'TFIDF': 'TFIDF',
    'TFIDF_NGRAM': 'TFIDF_NGRAM',
    'TFIDF_NGRAM_CHAR': 'TFIDF_NGRAM_CHAR',
    'EMBD': 'EMBD'
}

exp_options = {
    "E": "email_subject",
    "ED": "email_domains",
    "EQ": "email_quotes",
    "EQD": "email_quotes_domains",
    "B": "body"
}

NAME_CONNECTORS = ['in', 'the', 'all', 'for', 'and', 'on', 'but', 'at', 'of', 'to', 'a']
NAME_MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
               'september', 'october', 'november', 'december']
NAME_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']