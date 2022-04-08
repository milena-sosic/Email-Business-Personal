from nrclex import NRCLex
from textblob import TextBlob
from textstat import *
import re
import nltk
from nltk import word_tokenize
from pandas import read_pickle, read_csv
import numpy as np
import string
from constants import NAME_CONNECTORS, NAME_MONTHS, NAME_DAYS
from statistics import mean
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_emotional_features(data, experiment):
    content = data[experiment]

    def extract_emotions(text):
        text_object = NRCLex(text)
        af = text_object.affect_frequencies
        return af['fear'], af['anger'], af['trust'], af['surprise'], af['sadness'], \
               af['disgust'], af['joy'], af['positive'], af['negative']

    data['fear'], data['anger'], data['trust'], data['surprise'], data['sadness'], \
    data['disgust'], data['joy'], data['positive'],  data['negative'] = zip(
        *content.apply(lambda x: extract_emotions(x)))
    return


def create_lexical_features(data, experiment):
    content = data[experiment]
    subject = data['subject']

    def avg_sentence_length(text):
        words = textstat.lexicon_count(text, removepunct=True)
        sentences = textstat.sentence_count(text)
        average_sentence_length = float(words / sentences)
        return average_sentence_length

    def avg_syllables_per_word(text):
        syllable = textstat.syllable_count(text)
        words = textstat.lexicon_count(text, removepunct=True)
        aspw = float(syllable) / (float(words) + 1)
        return np.round(aspw, 1)

    def find_punctuations(text):
        remove = string.punctuation
        pattern = r"[{}]".format(remove)  # create the pattern
        punctuations_list = re.findall(pattern, text)
        return len(punctuations_list)

    def find_buss(text):
        with open('./lexicons/business_lexicon.txt', 'r') as f:
            f_content = f.read()
            buss_acr = re.split('\\n', f_content)
            buss_acr = [y for x in buss_acr for y in x.split(",")]

        tokens = nltk.tokenize.word_tokenize(text.lower())
        business_list = [word for word in tokens if word in buss_acr]
        print(business_list)
        return len(business_list)/len(tokens)

    def find_abrs(text):
        with open('./lexicons/business_acronyms.txt', 'r') as f:
            f_content = f.read()
            buss_acr = re.split('\\n', f_content)
            buss_acr = [x.split(";")[0] for x in buss_acr]

        buss_reg = re.findall(r"\b[A-Z\.]{2,}s?\b", text)
        buss_acr += buss_reg
        tokens = nltk.tokenize.word_tokenize(text)
        acronyms_list = [word for word in tokens if word in buss_acr]
        print(acronyms_list)
        return len(acronyms_list)/(len(tokens) + 1)

    data['acronyms_indicator'] = content.apply(lambda x: find_abrs(x))
    data['business_indicator'] = content.apply(lambda x: find_buss(x))

    data['content_lex_count'] = content.apply(lambda x: textstat.lexicon_count(x, removepunct=True))
    data['content_lex_length'] = content.apply(lambda x: len(x))
    data['subject_lex_count'] = subject.apply(lambda x: textstat.lexicon_count(x, removepunct=True))
    data['subject_lex_length'] = subject.apply(lambda x: len(x))
    data['newlines'] = content.apply(lambda x: x.count('\n'))
    data['spaces'] = content.apply(lambda x: sum(not chr.isspace() for chr in x))
    data['punctuations'] = content.apply(lambda x: find_punctuations(x))
    data['question_marks_ratio'] = content.apply(lambda x: x.count('?')) / (data['punctuations'] + 1)
    data['exclamation_marks_ratio'] = content.apply(lambda x: x.count('!')) / (data['punctuations'] + 1)
    data['dots_ratio'] = content.apply(lambda x: x.count('.')) / (data['punctuations'] + 1)
    data['hash_tags_ratio'] = content.apply(lambda x: x.count('#')) / (data['punctuations'] + 1)
    data['ref_tags_ratio'] = content.apply(lambda x: x.count('@')) / (data['punctuations'] + 1)
    data['sentences'] = content.apply(lambda x: textstat.sentence_count(x))
    data['sentences_density'] = data['sentences'] / (data['newlines'] + 1)
    data['words_density'] = data['content_lex_count'] / (data['spaces'] + 1)
    data['avg_sentence_length'] = content.apply(lambda x: avg_sentence_length(x))
    data['syllable_count'] = content.apply(lambda x: textstat.syllable_count(x))
    data['avg_syll_per_sentence'] = data['syllable_count'] / (data['sentences'] + 1)
    data['avg_syll_per_word'] = content.apply(lambda x: avg_syllables_per_word(x))
    data['difficult_words'] = content.apply(lambda x: textstat.difficult_words(x))
    data['difficult_words_ratio'] = data['difficult_words'] / (data['content_lex_count'] + 1)
    return


def create_conversational_features(data):
    content_body = data['body']
    content_splitters = data['splitter_content']

    def get_free_domains(text):
        with open('../lexicons/free_domains.txt', 'r') as f:
            f_content = f.read()
            processed = re.split('\n', f_content)
        processed = [x for x in processed if x]
        free_domains = [domain for domain in processed if (domain in text)]
        return free_domains

    def get_domains(text):
        domains = re.findall(r"[@]\w+[.]\w{2,3}", text)
        domains = [x[1:].lower() for x in domains if x]
        print(domains)
        free_domains = get_free_domains(text)
        return len(free_domains)/(len(domains) + 1), " ".join(domains + free_domains)

    data['free_domains_body_ratio'], data['domains_in_body'] = \
        zip(*content_body.apply(lambda x: get_domains(x)))
    data['free_domains_splitters_ratio'], data['domains_in_splitters'] = \
        zip(*content_splitters.apply(lambda x: get_domains(x)))
    return


def create_readability_features(data, experiment):
    content = data[experiment]
    data['flesch_reading_ease'] = content.apply(lambda x: textstat.flesch_reading_ease(x))
    data['automated_readability_index'] = content.apply(lambda x: textstat.automated_readability_index(x))
    data['linsear_write_formula'] = content.apply(lambda x: textstat.linsear_write_formula(x))
    return


def create_expressions_features(data, experiment):
    content = data[experiment]

    def extrat_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity, \
               len(blob.noun_phrases), len(blob.noun_phrases) / (textstat.lexicon_count(text, removepunct=True) + 1)

    data['polarity'], data['subjectivity'], data['noun_phrases'], data['noun_phrases_ratio'] = \
        zip(*content.apply(lambda x: extrat_sentiment(x)))
    return


def create_morality_features(data, experiment):
    content = data[experiment]

    # Load E-MFD
    emfd = read_csv('./lexicons/morality/emfd_scoring.csv', index_col='word')
    probabilites = [c for c in emfd.columns if c.endswith('_p')]
    foundations = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    senti = [c for c in emfd.columns if c.endswith('_sent')]
    emfd = emfd.T.to_dict()

    # Load eMFD-all-vice-virtue
    emfd_all_vice_virtue = read_pickle('./lexicons/morality/emfd_all_vice_virtue.pkl')

    # Load eMFD-single-vice-virtue
    emfd_single_vice_virtue = read_pickle('./lexicons/morality/emfd_single_vice_virtue.pkl')

    # Load eMFD-single-sent
    emfd_single_sent = read_pickle('./lexicons/morality/emfd_single_sent.pkl')

    def tokenizer(doc):
        return [x.lower_ for x in doc if
                not x.is_punct and not x.is_digit and not x.is_quote and not x.like_num and not x.is_space]

    def score_emfd_all_sent(text):

        # Initiate dictionary to store scores
        emfd_score = {k: 0 for k in probabilites + senti}
        text_tokenized = tokenizer(text)

        # Collect e-MFD data for all moral words in document
        moral_words = [emfd[token] for token in text_tokenized if token in emfd.keys()]
        print(len([token for token in text_tokenized if token in emfd.keys()]))
        print(moral_words)
        for dic in moral_words:
            emfd_score['care_p'] += dic['care_p']
            emfd_score['fairness_p'] += dic['fairness_p']
            emfd_score['loyalty_p'] += dic['loyalty_p']
            emfd_score['authority_p'] += dic['authority_p']
            emfd_score['sanctity_p'] += dic['sanctity_p']

            emfd_score['care_sent'] += dic['care_sent']
            emfd_score['fairness_sent'] += dic['fairness_sent']
            emfd_score['loyalty_sent'] += dic['loyalty_sent']
            emfd_score['authority_sent'] += dic['authority_sent']
            emfd_score['sanctity_sent'] += dic['sanctity_sent']

        if len(moral_words) != 0:
            emfd_score = {k: v / len(moral_words) for k, v in emfd_score.items()}
            nonmoral_words = len(text_tokenized) - len(moral_words)
            try:
                emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
            except ZeroDivisionError:
                emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
        else:
            emfd_score = {k: 0 for k in probabilites + senti}
            nonmoral_words = len(text_tokenized) - len(moral_words)
            try:
                emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
            except ZeroDivisionError:
                emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1

        return emfd_score['care_p'], emfd_score['fairness_p'], emfd_score['loyalty_p'], emfd_score['authority_p'], \
               emfd_score['sanctity_p'], emfd_score['care_sent'], emfd_score['fairness_sent'], emfd_score[
                   'loyalty_sent'], emfd_score['authority_sent'], emfd_score['sanctity_sent'], emfd_score[
                   'moral_nonmoral_ratio']

    data['care_p'], data['fairness_p'], data['loyalty_p'], data['authority_p'], data['sanctity_p'], \
    data['care_sent'], data['fairness_sent'], data['loyalty_sent'], data['authority_sent'], data['sanctity_sent'], \
    data['moral_nonmoral_ratio'] = zip(*content.apply(lambda x: score_emfd_all_sent(x)))
    return


def create_ner_count_features(data, experiment):
    content = data[experiment]

    def extract_ner(sent, label='PERSON'):
        ner_list = []
        ner = []
        name = ""
        for subtree in sent.subtrees(filter=lambda t: t.label() == label):
            for leaf in subtree.leaves():
                ner.append(leaf[0])
            for part in ner:
                name += part + ' '
            if name[:-1] not in ner_list:
                ner_list.append(name[:-1])
            name = ''
            ner = []
        return ner_list

    def get_entities(text):
        tokens = nltk.tokenize.word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        sent = nltk.ne_chunk(pos, binary=False)

        person_list = extract_ner(sent, label='PERSON')

        with open('./lexicons/personal_names.txt', 'r') as f:
            f_content = f.read()
            personal_names_lexicon = re.split(', ', f_content)

        person_list += personal_names_lexicon

        organization_list = extract_ner(sent, label='ORGANIZATION')

        connectors = NAME_CONNECTORS
        months = NAME_MONTHS
        days = NAME_DAYS
        numbers = re.findall(r"\w*\d(?:-\d+)*\w*", text)
        links = re.findall(r"http\S+", text) + re.findall(r"bit.ly/\S+", text)
        emails = re.findall(r"[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}", text)

        tokens = word_tokenize(text)  # nltk
        avg_token_len = mean([len(token) for token in tokens])

        names_list = [word for word in tokens if word in person_list]
        orgs_list = [word for word in tokens if word in organization_list]
        conn_list = [word for word in tokens if word in connectors]
        months_list = [word for word in tokens if word in months]
        days_list = [word for word in tokens if word in days]
        numbers_list = [word for word in tokens if word in numbers]
        links_list = [word for word in tokens if word in links]
        emails_list = [word for word in tokens if word in emails]
        tok_len = len(tokens) + 1

        return avg_token_len, len(names_list) / tok_len, len(orgs_list) / tok_len, \
               len(numbers_list) / tok_len, len(conn_list) / tok_len, len(months_list) / tok_len, \
               len(days_list) / tok_len, len(links_list) / tok_len, len(emails_list) / tok_len

    data['avg_word_length'], data['names_lex_ratio'], data['orgs_lex_ratio'], data['numbers_lex_ratio'], \
    data['conns_lex_ratio'], data['months_lex_ratio'], data['days_lex_ratio'], \
    data['links_lex_ratio'], data['emails_lex_ratio'] = zip(*content.apply(lambda x: get_entities(x)))
    return
