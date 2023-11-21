import pandas as pd
import subprocess
from pathlib import Path
from collections import defaultdict

datasets = ["train-mixed-10h", "train-other-10h", "train-other-60h"]

def make_filename(dataset, field):
    suffix = field.replace('trans_', '')
    output_filename = f'data/{dataset}.{suffix}.tsv'
    return output_filename

max_wer = {}
argmax_wer = {}
dfs = {}

for dataset in datasets:
    processed_df = pd.read_csv(f'data/transcription/processed/{dataset}_processed.csv')

    #print(processed_df.columns)
    gt_field = 'trans_ground_truth'
    fields = ['trans_human_worst_before', 'trans_human_random_before']
    for field in fields + [gt_field]:
        try:
            df = processed_df[['utt_id', field]]
        except KeyError:
            print('no field', field, 'in', dataset)
            continue
        df.loc[:, field] = df[field].str.upper()
        output_filename = make_filename(dataset, field)
        print('writing', output_filename)
        df.to_csv(output_filename, sep='\t', index=False, header=False)
        df.loc[:, ['dataset']] = dataset
        dfs[(dataset, field)] = df.rename(columns={field: 'text'})

    for other_field in fields:
        if Path(make_filename(dataset, other_field)).exists():
            output = subprocess.check_output([
                'compute-wer',
                f'ark:{make_filename(dataset, gt_field)}',
                f'ark:{make_filename(dataset, other_field)}'
            ], text=True)
            print(output)
            wer = float(output.split('\n')[0].split()[1])
            max_wer[dataset] = max(max_wer.get(dataset, 0), wer)
            if wer >= max_wer[dataset]:
                argmax_wer[dataset] = (dataset, other_field)

order = sorted(argmax_wer, key=lambda x: max_wer[x], reverse=True)

joint_df = pd.concat([dfs[argmax_wer[dataset]] for dataset in order]).groupby('utt_id').first().reset_index()
gt_df = pd.concat([dfs[(dataset, 'trans_ground_truth')] for dataset in order]).groupby('utt_id').first().reset_index()

def make_filename(row):
    utt_id, dataset = row['utt_id'], row['dataset']
    book, chapter, utt = utt_id.split('_')
    utt = utt.zfill(4)
    path = Path(f'data/audio/{dataset}/{book}/{chapter}/{book}-{chapter}-{utt}.flac')
    assert path.exists(), path
    return path

with open(f'data/crowd1.dirty.tsv', 'w') as f:
    for i, row in joint_df.iterrows():
        print(make_filename(row), row['text'], sep='\t', file=f)

with open(f'data/crowd1.clean.tsv', 'w') as f:
    for i, row in gt_df.iterrows():
        print(make_filename(row), row['text'], sep='\t', file=f)
