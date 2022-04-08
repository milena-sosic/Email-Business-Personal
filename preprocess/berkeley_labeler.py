import pandas as pd
import numpy as np


# Convert from original enron data format to tabular format with appropriately assigned labels for processing
def label_berkeley_data(data):
    labeled_enron = data[data.labeled.eq(1)]
    labeled_enron.loc[:, 'Cat_1_level_1': 'Cat_12_weight'].fillna(0)

    labels = []

    for index, row in labeled_enron.iterrows():
        row_labels = [0] * 57

        for i in range(12):
            try:
                top_level = int(row['Cat_{}_level_1'.format(i + 1)])
            except ValueError:
                top_level = top_level
            try:
                sec_level = int(row['Cat_{}_level_2'.format(i + 1)])
            except ValueError:
                sec_level = 0

            if top_level == 1:
                row_labels[top_level - 1] = 1
                row_labels[4 + sec_level - 1] = 1
            elif top_level == 2:
                row_labels[top_level - 1] = 1
                row_labels[12 + sec_level - 1] = 1
            elif top_level == 3:
                row_labels[top_level - 1] = 1
                row_labels[25 + sec_level - 1] = 1
            elif top_level == 4:
                row_labels[top_level - 1] = 1
                row_labels[38 + sec_level - 1] = 1
            else:
                continue

        labels.append(row_labels)

    for i in range(12):
        del labeled_enron['Cat_{}_level_1'.format(i + 1)]
        del labeled_enron['Cat_{}_level_2'.format(i + 1)]
        del labeled_enron['Cat_{}_weight'.format(i + 1)]

    labeled_enron.drop(labeled_enron.columns[labeled_enron.columns.str.contains('unnamed', case=False)], axis=1,
                       inplace=True)

    columns = ["Top-Level:Coarse-Genre", "Top-Level:Incl/Fwd-Inf", "Top-Level:Primary-Topics",
               "Top-Level:Emotional-Tone",
               "CG-Business-Strategy", "CG-Purely-Pesonal", "CG-Personal+Prof", "CG-Logistics", "CG-Employment",
               "CG-Document", "CG-Empty-MissingAtt",
               "CG-Empty", "IFI-New+FRW", "IFI-FRW+Rpl", "IFI-Formal-Docs", "IFI-News-Article", "IFI-Gvmt/Acd-Report",
               "IFI-Gvmt-Action",
               "IFI-Press", "IFI-Legal-Docs", "IFI-URLs", "IFI-Newsletter", "IFI-Humor-Business", "IFI-Humor-Private",
               "IFI-Attachments",
               "PT-Regulations", "PT-Internal-Projects", "PT-Company-Image-Curr", "PT-Company-Image-Chg",
               "PT-Political-Influence",
               "PT-CA-Energy-Crisis", "PT-Internal-Policy", "PT-Internal-Ops", "PT-Alliances", "PT-Legal-Advice",
               "PT-Talking-Points",
               "PT-Meeting-Minutes", "PT-Trip-Reports", "ET_Jubilation", "ET-Hope", "ET-Humor", "ET-Camaraderie",
               "ET-Admiration", "ET-Gratitude",
               "ET-Friendship", "ET-Sympathy", "ET-Sarcasm", "ET-Confidential", "ET-Anxiety", "ET-Concern",
               "ET-Aggressive", "ET_Triumph",
               "ET-Pride", "ET-Anger", "ET-Sadness", "ET-Shame", "ET-Dislike"]

    df_labels = pd.DataFrame(labels)
    df_labels.columns = columns

    data = pd.DataFrame(np.column_stack([labeled_enron, df_labels]),
                        columns=labeled_enron.columns.append(df_labels.columns))
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    data.to_csv("./data/berkeley_enron.csv", sep='\t', encoding='utf8')
    return data


def final_label(data):
    data['email_final_label'] = np.max(data['CG-Purely-Pesonal'].to_numpy(),
                                       data['CG-Personal+Prof'].to_numpy(),
                                       data["IFI-Humor-Private"].to_numpy())
    return
