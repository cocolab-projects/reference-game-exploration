import glob
import json
import os
import numpy as np
import pandas as pd
import re

class ReferenceDataset():
    """ Generate datasets for Reference games.
    """
    @staticmethod
    def construct_dataset(data_dir):
        """ Construct dataset of concatenated informative message, target (image),
            distractor 1 (image), distractor 2 (image) and 2 outputs (gold truth, listener selection).
        """
        dirs = ['../data/{}/train'.format(data_dir), '../data/{}/val'.format(data_dir), '../data/{}/test'.format(data_dir)]
        for d in dirs:
            msgs = pd.read_csv(os.path.join(d, 'msgs.tsv'), sep='\t')
            msgs = msgs[msgs["sender"]=="speaker"]
            msgs['example_id'] = msgs.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
            group = msgs.groupby(['example_id'])
            concat_msgs = pd.DataFrame(group['message'].apply(' '.join))
            concat_msgs = concat_msgs.reset_index()

            # Produce table of stimuli responses for a specific stimuli and conversation.
            responses = pd.read_csv(os.path.join(d, 'responses.tsv'), sep='\t')[['trialNum', 'gameid', 'selection']]
            responses['example_id'] = responses.apply(lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)

            # Combine stimuli responses with msgs
            dataset = responses.merge(concat_msgs)

            # Combine with stimuli ids
            stim_ids = []
            stim_id_files = glob.glob('../data/{}/raw/vision/ids/*.json'.format(data_dir))
            for f in stim_id_files:
                x = pd.read_json(f).transpose()
                x['trialNum'] = x.index
                x['gameid'] = os.path.splitext(os.path.split(f)[1])[0]
                x['example_id'] = x.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
                for c in ['distr1', 'distr2', 'target']:
                    x[c] =[re.sub(r'.svg', '.png', img_name) for img_name in x[c].tolist()]
                stim_ids.append(x)
            # Drop examples with incorrect answers
            stim_ids = pd.concat(stim_ids)
            dataset = dataset.merge(stim_ids)
            dataset = dataset.loc[dataset['selection'] == 'target']
            dataset = dataset.drop(columns=['selection', 'gameid', 'trialNum'])
            dataset.to_csv(os.path.join(d, 'vision', 'dataset.tsv'), sep='\t', index=False)


def main():
    # concept_dataset = ConceptDataset()
    # concept_dataset.construct_dataset()
    ref_dataset = ReferenceDataset()
    ref_dataset.construct_dataset('pilot_coll1')

if __name__ == '__main__':
    main()
