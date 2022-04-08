from tensorflow.python.keras.layers import Bidirectional
from modeling.generator import build_model_nn
from constants import *
import tensorflow_hub as hub
from keras.layers import dot
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D
import tensorflow as tf
from keras import regularizers
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import concatenate
from text.bert_embeddings import create_bert_embeddings
from utils import *


"""## CNN Neural Network"""
def cnn_neural_network(max_len=128, meta_len=0, experiment='E', use_meta=False):

    text_input = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    ]
    if use_meta:
        meta_input = Input(shape=(meta_len,), name='meta_input')

    embeddings, vocab_size, embeddings_len = create_bert_embeddings(text_input,
                                                                    seq_length=max_len, type='WORD',
                                                                    mode='SYMB', experiment=experiment)

    nlp_out = Conv1D(NUM_FILTERS, 7, activation='relu', padding='same')(embeddings)
    nlp_out = MaxPooling1D(2)(nlp_out)
    nlp_out = Conv1D(NUM_FILTERS, 7, activation='relu', padding='same')(nlp_out)
    nlp_out = GlobalMaxPooling1D()(nlp_out)
    nlp_out = Dropout(0.4)(nlp_out)
    if use_meta:
        x = concatenate([nlp_out, meta_input])
    else:
        x = nlp_out
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    if use_meta:
        inputs = [text_input, meta_input]
    else:
        inputs = text_input
    model = Model(inputs=inputs, outputs=[x])
    return model, vocab_size, embeddings_len


def cnn(train_data, train_y, test_data, test_y, experiment='E', use_meta=False):
    meta_len = 0
    if use_meta:
        meta_len = train_data[1].shape[1]
    print(meta_len)
    cnn_model, vocab_size, embeddings_len = cnn_neural_network(max_len=MAX_SEQ_LEN, meta_len=meta_len,
                                                      experiment=experiment,use_meta=use_meta)
    meta = '_META' if use_meta else ''
    result = build_model_nn(cnn_model, train_data, train_y, test_data, test_y, name=f"CNN_BERT{meta}",
                            scoring=score_metrics, n_splits=FOLDS, save=save_model,
                            epochs=EPOCHS, vocab_size=vocab_size, batch_size=BATCH_SIZE,
                            max_seq_len=MAX_SEQ_LEN, embeddings_len=embeddings_len, experiment=experiment
                            )
    return result


"""## BiLSTM - Embedding"""
def bilstm_neural_network(max_len=128, meta_len=0, experiment='E', use_meta=False):
    text_input = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    ]
    if use_meta:
        meta_input = Input(shape=(meta_len,), name='meta_input')

    embeddings, vocab_size, embeddings_len = create_bert_embeddings(text_input,
                                                                    seq_length=max_len, type='WORD',
                                                                    mode='SYMB', experiment=experiment)
    nlp_out = Bidirectional(LSTM(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001)))(embeddings)
    if use_meta:
        x = concatenate([nlp_out, meta_input])
    else:
        x = nlp_out
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    if use_meta:
        inputs = [text_input, meta_input]
    else:
        inputs = text_input
    model = Model(inputs=inputs, outputs=[x])
    return model, vocab_size, embeddings_len


def bilstm(train_data, train_y, test_data, test_y, experiment='E', use_meta=False):
    meta_len = 0
    if use_meta:
        meta_len = train_data[1].shape[1]
    print(meta_len)

    bilstm_model, vocab_size, embeddings_len = bilstm_neural_network(max_len=MAX_SEQ_LEN, meta_len=meta_len,
                                                      experiment=experiment, use_meta=use_meta)
    meta = '_META' if use_meta else ''
    result = build_model_nn(bilstm_model, train_data, train_y, test_data, test_y, name=f"BILSTM_BERT{meta}",
                            scoring=score_metrics, n_splits=FOLDS, save=save_model,
                            epochs=EPOCHS, vocab_size=vocab_size, batch_size=BATCH_SIZE,
                            max_seq_len=MAX_SEQ_LEN, embeddings_len=embeddings_len, experiment=experiment)
    return result


"""## BiLSTM + Attention - Embeddings + Meta"""
SINGLE_ATTENTION_VECTOR = True
APPLY_ATTENTION_BEFORE_LSTM = False
TIME_STEPS = 128


def attention_3d_block(hidden_states, max_len):
    hidden_size = int(hidden_states.shape[2])

    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)

    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(max_len, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


def UniversalEmbedding(x):
    embed = hub.load(BERT_MODEL)
    return embed.signatures['default'](tf.constant(x))


def bilstm_attention_network(max_len=128, meta_len=0, experiment='E', use_meta=False):
    text_input = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    ]
    if use_meta:
        meta_input = Input(shape=(meta_len,), name='meta_input')

    embeddings, vocab_size, embeddings_len = create_bert_embeddings(text_input,
                                                                    seq_length=max_len, type='WORD',
                                                                    mode='SYMB', experiment=experiment)

    nlp_out = Bidirectional(LSTM(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001)))(embeddings)
    attention_mul = attention_3d_block(nlp_out, max_len)

    if use_meta:
        x = concatenate([attention_mul, meta_input])
    else:
        x = attention_mul
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    if use_meta:
        inputs = [text_input, meta_input]
    else:
        inputs = text_input

    model = Model(inputs=inputs, outputs=[x])
    return model, vocab_size, embeddings_len


def bilstm_attention(train_data, train_y, test_data, test_y, experiment='E', use_meta=False):
    meta_len = 0
    if use_meta:
        meta_len = train_data[1].shape[1]
    bilstm_attention_model, vocab_size, embeddings_len = bilstm_attention_network(max_len=MAX_SEQ_LEN, meta_len=meta_len,
                                                      experiment=experiment, use_meta=use_meta)
    meta = '_META' if use_meta else ''
    result = build_model_nn(bilstm_attention_model, train_data, train_y, test_data, test_y,
                            name=f"BILSTM_ATT_BERT{meta}", scoring=score_metrics, n_splits=FOLDS,
                            save=save_model, epochs=EPOCHS, vocab_size=vocab_size, batch_size=BATCH_SIZE,
                            max_seq_len=MAX_SEQ_LEN, embeddings_len=embeddings_len, experiment=experiment)
    return result

