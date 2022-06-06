"""
File containing simple utils for data manipulation.
"""

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold


def stratified_k_fold_split(
    X,
    y=None,
    n_folds=1,
    test_size=0.2,
    random_state=None,
    regression=False,
):
    # Helper method to return a list with n_folds tuples (X_train, y_train)
    # after partitioning the given X, y dataset using stratified splits.
    result = []
    if y is None:
        # Then simply subsample X
        num_samples = np.floor(test_size * X.shape[0])
        selected_rows = np.random.choice(
            X.shape[0],
            size=num_samples,
            replace=False,
        )
        return X[selected_rows, :]

    if (n_folds == 1) or regression:
        # Degenerate case: let's just dump all our indices as our single fold
        split_gen = ShuffleSplit(
            n_splits=n_folds,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        # Split data
        split_gen = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )

    # Save indices
    for train_indices, test_indices in split_gen.split(X, y):
        result.append((train_indices, test_indices))

    return result
