import scattertext as st
import spacy
import pandas as pd
from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, \
        produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer


def plot_characteristics(df):
    df["personal"] = df["email_final_label"].astype(str)
    df = df.assign(
        parse=lambda df: df.content.apply(st.whitespace_nlp_with_sentences)
    )
    corpus = st.CorpusFromParsedDocuments(
        df, category_col='personal', parsed_col='parse'
    ).build().get_unigram_corpus().compact(st.AssociationCompactor(2000))

    html = st.produce_scattertext_explorer(
        corpus,
        category='1', category_name='Personal', not_category_name='Business',
        minimum_term_frequency=0, pmi_threshold_coefficient=0,
        width_in_pixels=1000, metadata=corpus.get_df()['email_final_label'],
        transform=st.Scalers.dense_rank
    )
    open('./compact_chart.html', 'w').write(html)

    html = st.produce_characteristic_explorer(
        corpus,
        category='1',
        category_name='Personal',
        not_category_name='Business',
        metadata=corpus.get_df()['email_final_label']
    )
    open('ber_characteristic_chart.html', 'wb').write(html.encode('utf-8'))
    print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))

    html = st.produce_scattertext_explorer(corpus,
    category = '1',
    category_name = 'Personal',
    not_category_name = 'Business',
    width_in_pixels = 1000,
    metadata=corpus.get_df()['email_final_label'])
    open("email-visualization.html", 'wb').write(html.encode('utf-8'))

    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['Personal Score'] = corpus.get_scaled_f_scores('1')
    print(list(term_freq_df.sort_values(by='Personal Score', ascending=False).index[:10]))

    term_freq_df['Business Score'] = corpus.get_scaled_f_scores('0')
    print(list(term_freq_df.sort_values(by='Business Score', ascending=False).index[:10]))

    import IPython
    IPython.display.HTML(filename='./characteristic_chart.html')


def moral_foundations(df):
    df["personal"] = df["email_final_label"].astype(str)
    df = df.assign(
        parse=lambda df: df.content.apply(st.whitespace_nlp_with_sentences)
    )
    moral_foundations_feats = st.FeatsFromMoralFoundationsDictionary()
    corpus = st.CorpusFromPandas(df,
                                 category_col='personal',
                                 text_col='content',
                                 nlp=st.whitespace_nlp_with_sentences,
                                 feats_from_spacy_doc=moral_foundations_feats).build()

    cohens_d_scorer = st.CohensD(corpus).use_metadata()
    term_scorer = cohens_d_scorer.set_categories('0', ['1']).get_score_df()

    print(term_scorer)

    html = st.produce_frequency_explorer(
        corpus,
        category='1',
        category_name='Personal',
        not_category_name='Business',
        use_non_text_features=True,
        use_full_doc=True,
        term_scorer=st.CohensD(corpus).use_metadata(),
        grey_threshold=0,
        width_in_pixels=1000,
        topic_model_term_lists=moral_foundations_feats.get_top_model_term_lists(),
        metadata_descriptions=moral_foundations_feats.get_definitions()
    )
    open('moral_foundations_chart.html', 'wb').write(html.encode('utf-8'))


def plot_phrases(df):

    df = df[~df.content_clean.isna()]
    df["personal"] = df["email_final_label"].astype(str)

    corpus = (CorpusFromPandas(df,
                               category_col='personal',
                               text_col='content',
                               feats_from_spacy_doc=PhraseMachinePhrases(),
                               nlp=spacy.load('en', parser=False))
              .build().compact(AssociationCompactor(4000)))

    html = produce_scattertext_explorer(corpus,
                                        category='1',
                                        category_name='Personal',
                                        not_category_name='Business',
                                        minimum_term_frequency=0,
                                        pmi_threshold_coefficient=0,
                                        transform=dense_rank,
                                        term_scorer=RankDifference(),
                                        width_in_pixels=1000)
    open('phrases_chart.html', 'wb').write(html.encode('utf-8'))


def plot_empath(df):
    df["personal"] = df["email_final_label"].astype(str)
    df = df.assign(
        parse=lambda df: df.content.apply(st.whitespace_nlp_with_sentences)
    )
    feat_builder = st.FeatsFromOnlyEmpath()
    empath_corpus = st.CorpusFromParsedDocuments(df,
        category_col = 'personal',
        feats_from_spacy_doc = feat_builder,
        parsed_col='parse').build()

    html = st.produce_scattertext_explorer(empath_corpus,
        category = '1',
        category_name = 'Personal',
        not_category_name = 'Business',
        width_in_pixels = 1000,
        use_non_text_features = True,
        use_full_doc = True,
        topic_model_term_lists = feat_builder.get_top_model_term_lists())
    open("empath_chart.html", 'wb').write(html.encode('utf-8'))


def plot_word_vectors(df, experiment):
    df["personal"] = df["email_final_label"].astype(str)
    df = df.assign(
        parse=lambda df: df[experiment].apply(st.whitespace_nlp_with_sentences)
    )
    corpus = (st.CorpusFromParsedDocuments(df,
                                           category_col='personal',
                                           parsed_col='parse').build()
              .get_stoplisted_unigram_corpus()
              .remove_infrequent_words(minimum_term_count=2, term_ranker=st.OncePerDocFrequencyRanker))

    embeddings = TfidfTransformer().fit_transform(corpus.get_term_doc_mat().T)

    print(embeddings.shape)
    print(corpus.get_num_docs())
    print(corpus.get_num_terms())

    U, S, VT = svds(embeddings, k=5, maxiter=5000, which='LM')
    print(U.shape)
    print(S.shape)
    print(VT.shape)
    x_dim = 0
    y_dim = 1
    projection = pd.DataFrame({'term': corpus.get_terms(),
                               'x': U.T[x_dim],
                               'y': U.T[y_dim]}).set_index('term')


    html = st.produce_pca_explorer(corpus,
                                   category='1',
                                   category_name='Personal',
                                   not_category_name='Business',
                                   projection=projection,
                                   width_in_pixels=1000,
                                   scaler=st.scale_neg_1_to_1_with_zero_mean,
                                   x_dim=x_dim,
                                   y_dim=y_dim)
    open("word_vectors_chart.html", 'wb').write(html.encode('utf-8'))

