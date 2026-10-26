import os
import dill

DATASET_CONFIG = {
    'mimic3': {'dir': 'output'},
    'mimic4': {'dir': 'mimic4'},
}


def transfer_procedure(data, diag_voc):
    """Shift procedure code indices by the diagnosis vocabulary size (+1 padding slot)."""
    offset = len(diag_voc.idx2word) + 1
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j][1])):
                data[i][j][1][k] += offset
    return data


def load_dataset(dataset='mimic3', data_dir='data'):
    """Load a preprocessed dataset.

    Each patient is a list of visits. Each visit is a list:
        [diag_codes, proc_codes, med_codes, age, gender]
    where diag_codes / proc_codes / med_codes are lists of integer indices,
    age is a float, and gender is an integer (0/1).

    Returns a dict with keys:
        train, eval, test, voc_size, ehr_adj, ddi_adj, ddi_path
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(f'Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIG)}.')
    base = os.path.join(data_dir, DATASET_CONFIG[dataset]['dir'])

    voc = dill.load(open(os.path.join(base, 'voc_final.pkl'), 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    records_path = os.path.join(base, 'records_final_hhh.pkl')
    if not os.path.exists(records_path):
        records_path = os.path.join(base, 'records_final.pkl')
    feature = dill.load(open(records_path, 'rb'))
    data = transfer_procedure(feature, diag_voc)

    ehr_adj = dill.load(open(os.path.join(base, 'ehr_adj_final.pkl'), 'rb'))
    ddi_adj = dill.load(open(os.path.join(base, 'ddi_A_final.pkl'), 'rb'))
    ddi_path = os.path.join(base, 'ddi_A_final.pkl')

    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)
    data_train = data[:split_point]
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    return {
        'train': data_train,
        'eval': data_eval,
        'test': data_test,
        'voc_size': voc_size,
        'ehr_adj': ehr_adj,
        'ddi_adj': ddi_adj,
        'ddi_path': ddi_path,
    }