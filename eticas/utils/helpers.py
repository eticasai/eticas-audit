import numpy as np


def binarize_predictions(scores):
    """
    Converts a list/array of continuous scores into binary predictions (0/1)
    using the mean.
    """
    mean_value = np.mean(scores)
    return np.where(scores > mean_value, 1, 0)


def get_mask(input_data, filters):
    """
    Get boolean array to filter data base on filters: Name of column and Conditions.
    {
        "name": "ethnicity",
        "privileged": [1]
    }
    """
    mask_privileged = np.ones(input_data.shape[0], dtype=bool)
    mask_underprivileged = np.ones(input_data.shape[0], dtype=bool)
    for filter in filters:

        if 'privileged' in filter:
            mask_underprivileged = mask_underprivileged & (
                ~input_data[filter['name']].isin(filter['privileged']))
            mask_privileged = mask_privileged & (
                input_data[filter['name']].isin(filter['privileged']))
        else:
            mask_underprivileged = mask_underprivileged & (
                input_data[filter['name']].isin(filter['underprivileged']))
            mask_privileged = mask_privileged & (
                ~input_data[filter['name']].isin(filter['underprivileged']))

    return mask_privileged, mask_underprivileged
