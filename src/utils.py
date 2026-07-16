## PACKAGES
import os, sys, warnings, pickle, multiprocess
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from itertools import permutations, combinations, product, islice, groupby
from skimage.measure import label, regionprops
from collections import Counter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import argrelextrema
from scipy.spatial import KDTree
import pandas as pd
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import poisson
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.stats import vonmises
from sklearn.linear_model import LinearRegression
from numpy.linalg import matrix_power

# import sparse
import re
import cv2

# from scipy.spatial.distance import jensenshannon

## SETTING
num_cpus = int(multiprocess.cpu_count() * 1)
np.set_printoptions(legacy="1.25")


## GENERAL MEASURES, SCORES
def fl(l):
    return [x for y in l for x in y]


def js_sim(p1, p2, axis=0):
    """
    NOTE:
        - JSS = 1-JSD which is bounded between 0 and 1 when using base 2
        - JSS: Jensen-Shannon Similarity
        - JSD: Jensen-Shannon Divergence
    """
    return 1 - jensenshannon(p1, p2, base=2, axis=axis)


def mutual_coverage(p1, p2, axis=-1):
    """
    NOTE:
    mutual coverages is defined as the product of
        1) how much p1 covers p2 and
        2) how much p2 covers p1
    """
    # assert (p1.sum(-1) <= 1 + 1e-6).prod() == 1
    # assert (p2.sum(-1) <= 1 + 1e-6).prod() == 1
    #
    cover_1 = (p1 * (p2 > 0)).sum(axis)
    cover_2 = ((p1 > 0) * p2).sum(axis)
    mc = cover_1 * cover_2
    return mc


def base_coverage(p_base, p, axis=-1):
    """
    NOTE:
    base coverages is defined as how much p_base covers p
    """
    assert (p_base.sum(-1) <= 1 + 1e-6).prod() == 1
    assert (p.sum(-1) <= 1 + 1e-6).prod() == 1
    #
    cover = (p * (p_base > 0)).sum(axis)
    return cover


def categorical_entropy(p, axis=-1, is_base_2=False):
    """use base 2 to compute how many bits given a discrete distribution"""
    assert np.isclose(np.sum(p, axis).mean(), 1)
    p = np.array(p)
    p += 1e-16
    if is_base_2:
        en = np.sum(-p * np.log2(p), axis)
    else:
        en = np.sum(-p * np.log(p) / np.log(len(p)), axis)
    return en


## UTILITY ALGORITHMS
def reduce_df_via_halving(df, attr, cumsum_frac, if_ascending):
    """
    NOTE:
    - the name halving here does not necessarily mean dividing by 2
    """
    df_sort = df.sort_values(attr, ascending=if_ascending)
    attr_sort = df_sort[attr].values
    attr_cumsum = attr_sort.cumsum() / attr_sort.sum()
    n_kept = np.where(attr_cumsum > cumsum_frac)[0][0]
    dff = df_sort[:n_kept]
    return dff


def reduce_rows_via_halving(rows, i_column, cumsum_frac, if_ascending):
    """
    NOTE:
    """
    # load
    row_arr = np.array(rows)
    column = row_arr[:, i_column]

    # sort
    if if_ascending:
        idx_sort = np.argsort(column)
    else:
        idx_sort = np.argsort(column)[::-1]
    row_arr_sort = row_arr[idx_sort]
    column_sort = column[idx_sort]

    # filter
    idx_removed = np.cumsum(column_sort) / column_sort.sum() > cumsum_frac
    idx_kept = ~idx_removed
    row_arr_kept = row_arr_sort[idx_kept]
    return row_arr_kept


def find_maxima_of_2d_array(z_array, maxima_mask=1.0, r_footprint=3, z_value_thres=0):
    """
    NOTE:
    - mask=1 for all elements is allowed to be a maximum
    - use the convention that flips x,y; i.e., axis_0:y, axis_1:x
    - keep only z_maxima > z_value_thres
    """
    z_masked = z_array * maxima_mask
    z_conv = maximum_filter(z_masked, footprint=np.ones([r_footprint, r_footprint]))
    detected_peaks = (z_conv == z_masked) & (z_masked > z_value_thres)
    y_maxima, x_maxima = np.where(detected_peaks)
    z_maxima = z_array[y_maxima, x_maxima]
    return x_maxima, y_maxima, z_maxima


def find_repeated_sequences(data, size):
    """
    Finds all repeated sequences of a specified size within a list.

    Args:
        data: The input list of integers.
        size: The size of the sequence to search for.

    Returns:
        A list of tuples, where each tuple contains a repeated sequence and a list of its starting indices.
    """
    if size > len(data) or size <= 0:
        return []

    sequences = {}
    for i in range(len(data) - size + 1):
        sequence = tuple(data[i : i + size])
        if sequence in sequences:
            sequences[sequence].append(i)
        else:
            sequences[sequence] = [i]

    repeated_sequences = [
        (seq, indices) for seq, indices in sequences.items() if len(indices) > 1
    ]
    return repeated_sequences


def get_ii_segm_from_bool_list(occ_list):
    bool_list = np.array(occ_list) > 0
    ts = [0] + np.cumsum([len(list(y)) for x, y in groupby(bool_list)]).tolist()
    tts = np.array(list(zip(ts[:-1], ts[1:])))
    bool_tt = [x == 1 for x, y in groupby(bool_list)]
    tts_r = tts[bool_tt]
    return tts_r
