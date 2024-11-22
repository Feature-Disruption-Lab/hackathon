"""Helper functions"""
from scipy import sparse
from typing import List, Dict

import numpy as np

def save_as_npz(data: List[Dict], filename: str):
    arrays_dict = {}
    for i, sample in enumerate(data):
        arrays_dict[f'prompt_{i}'] = sample['prompt']
        arrays_dict[f'response_{i}'] = sample['response']
        arrays_dict[f'token_indices_{i}'] = sample['token_indices']
        arrays_dict[f'feature_indices_{i}'] = sample['feature_indices']
        arrays_dict[f'magnitudes_{i}'] = sample['magnitudes']
        arrays_dict[f'label_{i}'] = sample['label']
    np.savez(filename, **arrays_dict)


def load_npz(filename: str):
    loaded = np.load(filename, allow_pickle=True)
    data = []
    i = 0
    while f'response_{i}' in loaded:

        sparse_matrix = sparse.coo_matrix((loaded[f'magnitudes_{i}'], (loaded[f'token_indices_{i}'], loaded[f'feature_indices_{i}'])))
        reconstructed = sparse_matrix.toarray()
        data.append({
            'prompt': str(loaded[f'prompt_{i}']),
            'response': str(loaded[f'response_{i}']),
            'features': reconstructed,
            'label': str(loaded[f'label_{i}']),
        })
        i += 1
    return data
