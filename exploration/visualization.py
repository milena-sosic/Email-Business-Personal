import os
from constants import root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, META_FEATURES, LABEL, LEX_FEATURES, MORAL_FEATURES, \
    NAME_FEATURE_DIR, NAME_MODEL_DIR, EMO_FEATURES
import seaborn as sns;
import matplotlib.pyplot as plt
import textstat
import numpy as np


def plot_sequence_length(data, experiment, perc_75, perc_95, upper_limit):
    content = data[experiment]
    content_lex_count = content.apply(lambda x: textstat.lexicon_count(x, removepunct=True))
    max_seq_len = np.round(content_lex_count.mean() + content_lex_count.std()).astype(int)

    kdeplot = sns.kdeplot(data=data[(data.content_lex_length < 1000)], x='content_lex_length', fill=True, common_norm=False, alpha=0.4)
    plt.axvline(x=128, color='k', linestyle='--', label='128 sequence length')
    plt.axvline(x=256, color='k', linestyle='--', label='256 sequence length')
    plt.axvline(x=512, color='k', linestyle='--', label='512 sequence length')
    plt.axvline(x=768, color='k', linestyle='--', label='768 sequence length')
    plt.axvline(x=perc_95, color='g', linestyle='--', label='95th percentile')
    plt.axvline(x=upper_limit, color='r', linestyle='--', label='Tukey\'s upper limit')

    x_ticks = np.append(plt.xticks()[0], 128)
    x_ticks = np.append(x_ticks, 256)
    x_ticks = np.append(x_ticks, 512)
    x_ticks = np.append(x_ticks, 768)
    x_ticks = np.append(x_ticks, perc_95)
    x_ticks = np.append(x_ticks, upper_limit)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.xticks(x_ticks)

    plt.legend(loc='upper right')
    plt.xlabel('Email Sequence Length')

    sns.despine()

    file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_FEATURE_DIR, f'SEQ_LEN_{experiment}')
    plt.savefig(f'{file_path}.png', bbox_inches='tight')
    plt.savefig(f'{file_path}.pdf', bbox_inches='tight')


def plot_features_distribution(data, experiment, train_id):
    label = data[LABEL].map({1: 'Personal', 0: 'Business'})
    lex_features = data[META_FEATURES]
    for feature in lex_features:
        plt.figure()
        kdeplot = sns.kdeplot(data=data, x=feature, hue=label, fill=True, common_norm=False, alpha=0.4)

        plt.legend(labels=['Business','Personal'], loc='best', frameon=False)

        file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_FEATURE_DIR,
                                 f'{train_id}_{feature}_{experiment}.png')
        sns.despine()
        kdeplot.figure.savefig(file_path, bbox_inches='tight')


def plot_features_importance(data, perm_data, type, experiment, train_id, date):
    plt.figure()
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = "8"

    barplot = sns.barplot(x=data['Importance'], y=data['Feature'], palette="Set2")
    plt.xlabel('Importance')
    plt.ylabel('')
    file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_FEATURE_DIR,
                             f'{train_id}_FI_BAR_{type}_{experiment}_{date}.png')
    plt.tight_layout()
    sns.despine()
    barplot.figure.savefig(file_path, bbox_inches='tight', palette="Set2")

    if type == 'PERM':
        plt.figure()
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = "8"
        boxplot = sns.boxplot(data=perm_data, orient='h', palette="Set2")
        plt.xlabel('Importance')
        plt.ylabel('')

        file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_FEATURE_DIR,
                                 f'{train_id}_FI_BOX_{type}_{experiment}_{date}.png')
        plt.tight_layout()
        sns.despine()
        boxplot.figure.savefig(file_path, bbox_inches='tight')


def plot_features_corr(X, hierarchy, corr, corr_linkage, experiment, train_id, date):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)
    file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_FEATURE_DIR,
                             f'{train_id}_CORR_{experiment}_{date}.png')
    fig.tight_layout()
    sns.despine()
    fig.savefig(file_path, bbox_inches='tight')


def plot_model_history(fit_history, plot_name):
    history_dict = fit_history.history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6), facecolor='#F0F0F0')
    fig.tight_layout()

    ax = plt.subplot(2, 1, 1)
    ax.set_facecolor('#F8F8F8')
    ax.plot(epochs, loss, label='Training')
    ax.plot(epochs, val_loss, label='Validation')
    ax.set_title('Model Accuracy & Loss')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = plt.subplot(2, 1, 2)
    ax.set_facecolor('#F8F8F8')
    ax.plot(epochs, acc, label='Training')
    ax.plot(epochs, val_acc, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')

    file_path = os.path.join(root_dir, NAME_RESULTS_DIR, NAME_PLOTS_DIR, NAME_MODEL_DIR, plot_name)
    sns.despine()
    plt.savefig(f'{file_path}.png', bbox_inches='tight')
    plt.savefig(f'{file_path}.pdf', bbox_inches='tight')
