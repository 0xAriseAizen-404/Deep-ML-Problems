import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """

    # indices = np.arange(n_samples)
    # if shuffle:
    #     np.random.shuffle(indices)
    # folds_lens = []
    # temp_nsamples = n_samples
    # temp_k = k
    # while temp_nsamples > 0:
    #     val = (temp_nsamples + temp_k - 1) // temp_k
    #     folds_lens.append(val)
    #     temp_nsamples -= val
    #     temp_k -= 1
    # split_indices = []
    # ind = 0
    # for val in folds_lens:
    #     split_indices.append([int(x) for x in indices[ind:ind + val]])
    #     ind += val
    
    # final_folds = []
    # for i in range(k):
    #     train_foldi = []
    #     for ind in range(len(split_indices)):
    #         if ind != i:
    #             train_foldi.extend(split_indices[ind])
    #     test_foldi = split_indices[i]
    #     final_folds.append((train_foldi, test_foldi))
    # return final_folds

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return [
        (
            [int(x) for j, fold in enumerate(folds) if j != i for x in fold],
            [int(x) for x in folds[i]]
        )
        for i in range(k)
    ]

# TC: O(n * k)
# SC: O(n * k)