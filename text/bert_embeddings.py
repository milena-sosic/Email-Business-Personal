import tensorflow_hub as hub
import tensorflow as tf
from constants import save_model, NAME_BERT_EMBEDDINGS, NAME_BERT_TOKENS, \
    NAME_MODEL_DIR, root_dir, BERT_MODEL, PREPROCESS_MODEL
import pickle
import os


def create_bert_embeddings(text_input, seq_length=128, type='WORD', mode='SYMB', experiment='E'):

    preprocessor = hub.load(PREPROCESS_MODEL)
    tokenize = hub.KerasLayer(preprocessor.tokenize)

    if mode == 'SYMB':
        tokenized_inputs = [tokenize(segment) for segment in text_input]
    else:
        tokenized_inputs = [tokenize(text_input)]

    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.

    encoder_input = bert_pack_inputs(tokenized_inputs)

    bert_model = hub.load(BERT_MODEL)

    encoder = hub.KerasLayer(bert_model, trainable=True)
    text_embeddings = encoder(encoder_input)

    pooled_output = text_embeddings["pooled_output"]  # [batch_size, embeddings_len].
    sequence_output = text_embeddings["sequence_output"]  # [batch_size, seq_length, embeddings_len].

    vocab_size = encoder.weights[0].shape[0]
    embeddings_len = encoder.weights[0].shape[1]
    if save_model:
        # save the model to disk
        filename = NAME_BERT_TOKENS + f'_{experiment}.sav'
        pickle.dump(tokenized_inputs, open(os.path.join(root_dir,
                                                  NAME_MODEL_DIR, 'vectorizers', filename), 'wb'))

        filename = NAME_BERT_EMBEDDINGS + f'_{experiment}.sav'
        pickle.dump(text_embeddings, open(os.path.join(root_dir,
                                               NAME_MODEL_DIR, 'vectorizers', filename), 'wb'))

    if type == 'WORD':
        return sequence_output, vocab_size, embeddings_len
    else:
        return pooled_output, vocab_size, embeddings_len


def create_sent_bert_embeddings(text_input, experiment='E'):
    encoder = hub.load(BERT_MODEL)
    text_embeddings = encoder.signatures['default'](tf.constant(text_input))

    vocab_size = encoder.weights[0].shape[0]
    embeddings_len = encoder.weights[0].shape[1]
    if save_model:
        # save the model to disk
        filename = NAME_BERT_EMBEDDINGS + f'_{experiment}.sav'
        pickle.dump(text_embeddings, open(os.path.join(root_dir,
                                                       NAME_MODEL_DIR, 'vectorizers', filename), 'wb'))

    return text_embeddings, vocab_size, embeddings_len


def extract_bert_model_parameters(model_name):
    model_name_parts = model_name.split("_")
    layers, hiddens, attentions = ''
    for part in model_name_parts:
        if 'L-' in part:
            layers = part.split("-")[-1]
        elif 'H-' in part:
            hiddens = part.split("-")[-1]
        elif 'A-' in part:
            attentions = part.split("-")[-1][0]
        else:
            continue
    return layers, hiddens, attentions

