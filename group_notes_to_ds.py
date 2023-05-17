import copy
import pandas as pd

new_entry={'src_notes': [], 'discharge_summary': None, 'index': None}
ds_grouped = [copy.deepcopy(new_entry)]
curr_index = 0
embedding_regex = "(?:\< ?(?:[A-z]| ?\/ ?)* ?\>)|\[ ?UNK ?\]"
df = pd.read_csv('path_to_Stockholm_EPR_Gastro ICD-10_Pseudo_Corpus_II_dataset')

# Preprocessing
# Drop all duplicate rows, and all rows with the same journalanteckning_id.
preprocessed_df = df.drop_duplicates().drop_duplicates(subset=['journalanteckning_id'])
preprocessed_df = preprocessed_df[preprocessed_df['full_note'] != '[UNK]'].copy() # Remove all rows consisting only of "[UNK]".
# Remove all embeddings and inline [UNK], strip trailing whitespaces.
preprocessed_df['full_note'] = preprocessed_df['full_note'].str.replace(embedding_regex, "").str.strip()
# Remove all rows whose notes are empty (in case this occurs as a result of removal of embeddings).
preprocessed_df = preprocessed_df[preprocessed_df['full_note'] != '']

for index, row in preprocessed_df.sort_values(by=["patientnr", "journalanteckning_id"]).reset_index().iterrows():
    if "epikris" in str(row['mall_namn']).lower():
        ds_grouped[curr_index]['discharge_summary'] = row
        ds_grouped[curr_index]['index'] = curr_index
        ds_grouped.append(copy.deepcopy(new_entry))

        # Filter out rows which have no discharge summary and thus were incorrectly assigned to this discharge summary.
        ds_grouped[curr_index]['src_notes'] = list(filter(
            lambda n: n['patientnr'] == ds_grouped[curr_index]['discharge_summary']['patientnr'],
            ds_grouped[curr_index]['src_notes']
        ))

        curr_index = curr_index+1
    else:
        ds_grouped[curr_index]['src_notes'].append(row)

# Filter out all groups which do not have source notes, or do not have a discharge summary.
ds_grouped = [g for g in ds_grouped if ((len(g['src_notes']) > 0) & (type(g['discharge_summary']) != None.__class__))]

# Concatenate all source notes for each DS and join together to a new dictionary.
finetune_set = list(map(lambda g: {'notes': " ".join(list(map(lambda n: n['full_note'], g['src_notes']))), 'summary': g['discharge_summary']['full_note']}, ds_grouped))