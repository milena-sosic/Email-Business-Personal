import nltk
from nltk.corpus import wordnet
import re
import numpy as np
import string
from constants import DEFAULT_MAIL_DOMAIN_ENRON
from preprocess.feature_extractor import create_conversational_features
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def prepare_email_content(data, train_data, normalize=False):

    if not normalize:
        if train_data == 'berkeley':
            data['subject'] = data['Subject']
            data['quotes'] = data['content']
            if not normalize:
                data['body'] = data['content']#.apply(lambda x: normalize_text(str(x), normalize).strip())
            data['splitter_list'] = data['content'].apply(lambda x: find_splitters_type(x))
            data['content'] = data["body"].apply(lambda x: delete_quotes(x).strip())
            data['recipients'] = data['To'].astype(str).map(
                lambda x: re.findall(r"[a-z0-9]+[\._-]?[a-z0-9]+[@]\w+[.]\w{2,3}", x))
            data['recipients_count'] = data['recipients'].apply(lambda x: len(x))
            data['to_domains'] = [set([x.split('@')[1] for x in l]) for l in data['recipients']]
            data['to_domains_count'], data['to_domains_type'] = \
                zip(*data.to_domains.apply(lambda x: check_domain_type(x)))
            data['splitter_content'] = data['body']
        else:
            data['content'] = data.apply(extract_content, axis=1)
            data["quotes"] = data.apply(extract_quote, axis=1)
            data['recipients_count'] = data['recipients'].apply(lambda x: len(x))
            data['email_depth'], data["splitter_list"], data['splitter_content'] = zip(*data.apply(extract_splitter, axis=1))
            data['from_domains_count'], data['to_domains_count'], data['to_domains_type'] = \
                zip(*data['header_info'].apply(lambda x: extract_to_domains(x)))

        create_conversational_features(data)
        # if train_data == 'berkeley':
        #     data['domains_in_splitters'] = data['domains_in_body']

    data['domains_in_splitters'] = data['domains_in_splitters'].replace(np.nan, '')
    data['splitter_list'] = data['splitter_list'].replace(np.nan, '')
    data['body'] = data['body'].apply(lambda x: normalize_text(str(x), normalize).strip())
    data["content_clean"] = data["content"].apply(lambda x: normalize_text(str(x), normalize).strip())
    data["subject_clean"] = data["subject"].apply(lambda x: normalize_text(str(x), normalize).strip())
    data["quotes_clean"] = data["quotes"].apply(lambda x: normalize_text(str(x), normalize).strip())
    data["domains_in_splitters_clean"] = data["domains_in_splitters"]\
        .apply(lambda x: normalize_text(str(x), normalize).strip())
    data["splitter_list_clean"] = data["splitter_list"].apply(lambda x: normalize_text(str(x), normalize).strip())

    data["email_subject"] = data['subject_clean'] + " " + data["content_clean"]
    data["email_domains"] = data["email_subject"] + " " + data["domains_in_splitters_clean"]
    data["email_domains_splitters"] = data["email_domains"] + " " + data["splitter_list_clean"]
    data['email_quotes'] = data['email_subject'] + " " + data['quotes_clean']
    data["email_quotes_domains"] = data["email_quotes"] + " " + data["domains_in_splitters_clean"]
    data["email_quotes_domains_splitters"] = data["email_quotes_domains"] + " " + data["splitter_list_clean"]
    if not normalize:
        data.to_csv(f'../data/{train_data}_enron_prepared.csv', sep='\t', encoding='utf8')

    return data


def find_splitters_type(text):

    text = str(text).lower()
    splitters = []
    re_list = re.findall("re:", text)
    re_list += re.findall("re;", text)

    fw_list = re.findall("fwd:", text)
    fw_list += re.findall("fw:", text)

    splitters = re_list + fw_list
    return " ".join(splitters)


def clean_data(data, experiment):
    data = data[~data[experiment].isna()]
    data = data[data[experiment] != '']
    data = data[data[experiment] != ' ']

    return data


def remove_links(text):
    '''Takes a string and removes web links from it'''
    text = re.sub(r'http\S+', ' ', text)  # remove http links
    text = re.sub(r'bit.ly/\S+', ' ', text)  # rempve bitly links
    text = text.strip(r'[link]')  # remove [links]
    return text


def remove_emails(text):
    text = re.sub(r'[a-z0-9]+[\._-]?[a-z0-9]+[@]\w+[.]\w{2,3}', ' ', text)  # remove email
    return text


def normalize_text(sentences, normalize=False):
    sent = sentences
    if normalize:
        sent = sent.lower()  # lower case
        sent = remove_emails(sent)
        sent = remove_links(sent)
        # sent = restore_contractions(sent)
        sent = re.sub(r'([0-9]+)', '', sent)  # remove numbers
        punctuations = string.punctuation
        pattern = r"[{}]".format(punctuations)  # create the pattern
        sent = re.sub(pattern, ' ', sent)
        sent = re.sub(r'\s+', ' ', sent)  # remove double spacing and newlines
        personal = []
        with open('./lexicons/personal_names.txt', 'r') as f:
            f_content = f.read()
            # splitting text.txt by non alphanumeric characters
            processed = re.split(', ', f_content)
            personal = processed
        for name in personal:
            if name in sent:
                sent = sent.replace(name, '')

    return sent


# Function to remove quotations from emails when the start end final
# positions of email content cannot be exactly determined
def delete_quotes(message):
    # nextMessage = re.split(r"\n.*[\,].*\<\s*.*>", message)[0]
    next_message = message.split(">-----")[0]
    next_message = next_message.split("----------------------")[0]
    next_message = next_message.split("-----")[0]
    next_message = next_message.split("To:")[0]
    next_message = next_message.split("From:")[0]
    return next_message


# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    # print(nltk_tag)
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence, lemmatizer):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            word_lemma = lemmatizer.lemmatize(word, tag)
            lemmatized_sentence.append(word_lemma)
    return " ".join(lemmatized_sentence)


def lemmatize_sentence_list(sentence, lemmatizer):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            word_lemma = lemmatizer.lemmatize(word, tag)
            lemmatized_sentence.append(word_lemma)
    return lemmatized_sentence


def consensus_label(text):
    labels = [int(i) for i in text.split(';')]
    iter = labels.copy()
    for i in iter:
        if i == 6:
            labels.remove(i)  # exclude 6s from calculation

    ties = len(labels) == len(set(labels))

    if len(labels) > 0:
        if not ties:
            label = max(labels, key=labels.count)
        else:
            label = int(np.floor(np.mean(labels)))
    else:
        label = 6
    return label


def binary_label(text):
    label = int(text)
    final_label = 6
    if (label == 3) or (label == 4) or (label == 5):  # personal
        final_label = 1
    elif (label == 1) or (label == 2):  # business
        final_label = 0
    return final_label


def restore_contractions(text):
    text = text.lower()
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"you'd", "you would", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"could've", "could have", text)
    text = re.sub(r"youve", "you have", text)
    return text


def create_dict(lst):
    d = dict()
    for item in lst:
        temp_list = item.strip("''").split(":")
        d[temp_list[0].strip('"')] = temp_list[1].strip('"')
    return d


def extract_to_domains(text):
    print(text)
    to_domains = []
    from_domains = []
    domain_type = 1

    if text != '':
        g = text.strip('[]').strip("''")
        g1 = g.strip("{}").split('},{')

        for item in g1:
            g2 = item.strip("'").split(",")
            print(g2)
            d = create_dict(g2)
            if d['role'] == 'to':
                to_domains.append(d['email_address'].lower().split("@", 1)[1])
            if d['role'] == 'from':
                from_domains.append(d['email_address'].lower().split("@", 1)[1])

        if (DEFAULT_MAIL_DOMAIN_ENRON in to_domains) and (len(set(to_domains)) == 1):
            domain_type = 1  # In
        elif (DEFAULT_MAIL_DOMAIN_ENRON in to_domains) and (len(set(to_domains)) > 1):
            domain_type = 2  # Mix
        else:
            domain_type = 3  # Out

    return len(from_domains), len(to_domains), domain_type


def extract_content(row):
    text = row['annotations']
    g = text.strip('[]').strip("''")
    g1 = g.strip("{}").split('},{')

    content = ''
    for item in g1:
        g2 = item.strip("'").split(",")
        d = create_dict(g2)
        print(d)
        for _ in d.keys():
            if d['annotation_type'] == 'OriginalMessageContent':
                start_index = int(d['start_index'])
                end_index = int(d['end_index'])
                print(row['body'])
                content = row['body'][start_index : end_index]
                break
    return content


def extract_quote(row):
    text = row['annotations']

    g = text.strip('[]').strip("''")
    g1 = g.strip("{}").split('},{')

    quotes = []
    for item in g1:
        g2 = item.strip("'").split(",")
        d = create_dict(g2)
        if d['annotation_type'] == 'MessageQuote':
            start_index = int(d['start_index'])
            end_index = int(d['end_index'])
            content = row['body'][start_index : end_index]
            quotes.append(content)
    return " ".join(quotes)


def extract_splitter(row):
    text = row['annotations']
    splitter_type = []
    splitter_text = []

    g = text.strip('[]').strip("''")
    g1 = g.strip("{}").split('},{')

    for item in g1:
        g2 = item.strip("'").split(",")
        d = create_dict(g2)
        print(d)
        if d['annotation_type'] == 'SplitterText':
            splitter_type.append(d['splitter_type'])
            start_index = int(d['start_index'])
            end_index = int(d['end_index'])
            content = row['body'][start_index : end_index]
            splitter_text.append(content)
    return len(splitter_type), " ".join(splitter_type), " ".join(splitter_text)


def check_domain_type(to_domains):
    domain_type = 1
    if (DEFAULT_MAIL_DOMAIN_ENRON in to_domains) and (len(set(to_domains)) == 1):
        domain_type = 1  # In
    elif (DEFAULT_MAIL_DOMAIN_ENRON in to_domains) and (len(set(to_domains)) > 1):
        domain_type = 2  # Mix
    else:
        domain_type = 3  # Out
    return len(to_domains), domain_type
