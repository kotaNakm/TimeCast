import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import json
import pickle
from itertools import product
from numpy.lib.stride_tricks import as_strided
from sklearn.preprocessing import scale,StandardScaler
from tqdm import trange
from scipy.fft import fft
from sklearn.cluster import KMeans
import dill
import random
import sys
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,normalized_mutual_info_score,homogeneity_score,mean_squared_log_error




sys.path.append("_dat")
from importlib import import_module

def root_mean_squared_percentage_error(actual,pred):
    return np.sqrt(np.mean(((pred - actual) / actual)**2))


def comment(strings):
    print("--------------------------------")
    print(strings)
    print("--------------------------------")


# can handle sequences with various length
def norm_list_matrix(data,scalers=None):
    # (# of seq, seq, features)
    comment("NORMALIZE LIST of MATRIX")
    if scalers is None:
        scalers=[]
        train=True
    else:
        train=False

    normed_data = [np.zeros(seq.shape) for seq in data]
    length_list = [seq.shape[0] for seq in data]

    n_features = data[0].shape[1]

    for d in range(n_features):
        d_temp = np.zeros(sum(length_list))
        st = 0
        for seq, len_ in zip(data, length_list):
            d_temp[st : st + len_] = seq[:, d]
            st += len_
        
        if train:
            scaler = StandardScaler()
            scaler.fit(d_temp.reshape(-1,1))
            scalers.append(scaler)
        else:
            pass

        normed_seq = scalers[d].transform(d_temp.reshape(-1,1))

        print(f"mean:{np.mean(d_temp),scalers[d].mean_} => {np.mean(normed_seq)}")
        print(f"std:{np.std(d_temp),np.sqrt(scalers[d].var_)} => {np.std(normed_seq)}")
        
        st = 0
        for seq_id, len_ in enumerate(length_list):
            normed_data[seq_id][:, d] = normed_seq[st : st + len_].reshape(-1)
            st += len_

    return normed_data, scalers


def import_dataset(args):
    """
    Return List of a DataFrame,
    which form of
    0| Dim1 | Dim2 | ... | Label |
    1| :
    2| :
    3| :
    """
    if args.input_tag == "Acute" or args.input_tag == "Chronic" or args.input_tag == "Mixed":
        dataset_module = import_module("mimic3-benchmarks-wrapper_time-to-event")
        assert  args.import_type == "clean_data_rul_k_folds"
        return dataset_module.load_clean_data_rul_k_folds(data=args.input_tag,split_ind = args.split_ind)

    elif args.input_tag == "Engine":
        dataset_module = import_module(args.input_tag)
        assert args.import_type == "clean_data_rul_k_folds_cmapss_condition1"
        return dataset_module.load_clean_data_rul_k_folds(args.split_ind,
            indices=["FD001","FD003",],use_test=False,
            )    
    elif args.input_tag == "Factory":
        assert args.import_type == "clean_data_rul_k_folds_once_per_machine"
        return dataset_module.load_clean_data_rul_k_folds(args.split_ind,
            once_per_machine_min=100,
            )

    else:
        NotImplementedError

def make_views(arr, window_size, step_size=1, writable=False):
    """
    arr: 2D-array
        e.g., [time x dim]

    https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
    """
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 1)

    # n_sample = ((len(arr) - window_size) // step_size)
    n_sample = ((len(arr) - window_size) // step_size) + 1

    n_dim = arr.shape[1]
    shape = (n_sample, window_size, n_dim)
    strides = (step_size * arr.itemsize, *arr.strides)

    return as_strided(arr, shape=shape, strides=strides)


def parse_columns(dataset, args):
    a_df = dataset[0]

    if args.label_col is None:
        raise ValueError("Label is needed")

    label_col = args.label_col
    if args.feature_cols is not None:
        feature_cols = args.feature_cols.split("/")

    if feature_cols is None:
        # Use all features
        target_cols = a_df.keys().to_list()
        feature_cols = a_df.drop(label_col, axis=1).keys().to_list()


    return label_col, feature_cols


def znorm_train_test_datasets(train_dataset,test_dataset,args):
    """
    Parameters
    ==========
    train_dataset:
    list of DataFrame    
    """
    label_col, feature_cols = parse_columns(train_dataset, args)
    normed_train_dataset, train_scalers = znorm_dataset(train_dataset,feature_cols)
    normed_test_dataset, _ = znorm_dataset(test_dataset,feature_cols,scalers=train_scalers)

    return normed_train_dataset, normed_test_dataset


def znorm_dataset(dataset,feature_cols,scalers=None):
    data_arr_list = [df[feature_cols].to_numpy() for df in dataset]

    data_arr_list, scalers = norm_list_matrix(data_arr_list,scalers=scalers)
    
    for m_id, normed_feature in enumerate(data_arr_list):
        dataset[m_id][feature_cols] = normed_feature
    
    return dataset,scalers


def make_samples(dataset, args, verbose=True):
    label_col, feature_cols = parse_columns(dataset, args)
    print(label_col, feature_cols)

    # # if znorm:
    # dataset = znorm_dataset(dataset,feature_cols)

    result_df = pd.DataFrame()

    features_set = []
    labels_set = []
    lastflag_set = []
    machine_id_set = []

    if args.use_dict_inputs:
        return _make_sample_dict(dataset, args, verbose)

    for machine_id, df_ in enumerate(dataset):
        if len(df_) < args.window_size:
            continue
        df = copy.deepcopy(df_)
        view_features = make_views(
            df[feature_cols].to_numpy(), args.window_size, args.step_size
        )
        features_set.append(view_features)

        # labels = df[label_col].to_numpy()[args.window_size-1:]
        # labels = np.expand_dims(labels,1)
        view_label = make_views(
            df[label_col].to_numpy(), args.window_size, args.step_size
        )
        # labels = view_label[:, 0]
        labels = view_label[:, -1]
        labels_set.append(labels)

        laflags = np.zeros(len(labels))
        laflags[-1] = 1
        lastflag_set.append(laflags)

        machine_id_set.append(np.full(len(labels), machine_id))

    all_features = np.concatenate(features_set, axis=0)
    all_labels = np.concatenate(labels_set, axis=0)
    all_lastflags = np.concatenate(lastflag_set, axis=0)
    all_ids = np.concatenate(machine_id_set, axis=0)

    result_df["seq_id"] = all_ids
    result_df["last_flag"] = all_lastflags
    result_df["actual"] = all_labels

    if verbose:
        print("TOTAL SAMPLE SIZE")
        print("=================")
        print("* features", all_features.shape)
        print("* label", all_labels.shape)
    

    if args.use_sequence_inputs:
        return (
            dict(
                features=features_set,
                label=labels_set,
                n_labels=len(np.unique(all_labels)),
                n_seq=len(dataset),
            ),
            result_df,
        )

    else:
        return (
            dict(
                features=all_features,
                label=all_labels,
                n_labels=len(np.unique(all_labels)),
                n_seq=len(dataset),
            ),
            result_df,
        )


def _make_sample_dict(dataset, args, verbose=True):
    """Generate a dict object containing input for Keras models.
    handle each features as dict
    """

    inputs = dict()
    result_df = pd.DataFrame()

    # add tensor info
    inputs.update({"n_seq": len(dataset)})

    # # add feature info
    # if args.feature_ids is None:
    #     raise ValueError("feature ids is needed to make an input dict")
    # else:
    #     feature_ids = args.feature_ids.split("/")

    label_col, feature_ids = parse_columns(dataset, args)

    inputs.update({"feature_ids": feature_ids})
    print("Feature IDs", feature_ids)

    features_set = [[] for _ in feature_ids]
    labels_set = []
    lastflag_set = []
    machine_id_set = []

    for machine_id, df_ in enumerate(dataset):
        if len(df_) < args.window_size:
            continue

        df = copy.deepcopy(df_)
        # Make labels
        view_label = make_views(
            df[label_col].to_numpy(), args.window_size, args.step_size
        )
        labels = view_label[:, -1]
        # get the latest label in each window
        labels_set.append(labels)

        # Make input features
        for f_id, feature_id in enumerate(feature_ids):
            # target_col_bool = data.columns.str.contains(pat=feature_id)
            # target_col_name = data.columns[target_col_bool]
            target_data = df[feature_id]
            target_view = make_views(
                target_data.to_numpy(), args.window_size, args.step_size
            )

            if args.use_fft:
                n_sample = len(target_view)
                samples = [None] * n_sample

                for i in range(n_sample):
                    samples[i] = compute_fft(target_view[i], tau=args.use_fft)

                # Add FFT Inputs
                features_set[f_id].append(np.array(samples))

            # elif args.use_wavelet:
            # pass
            # Add Wavelet Inputs
            else:
                # Add typical Inputs
                features_set[f_id].append(np.array(target_view))

        laflags = np.zeros(len(labels))
        laflags[-1] = 1
        lastflag_set.append(laflags)

        machine_id_set.append(np.full(len(labels), machine_id))


    all_features_each = [
        np.concatenate(features_set[f_id], axis=0)
        for f_id, feature_id in enumerate(feature_ids)
    ]
    all_labels = np.concatenate(labels_set, axis=0)
    all_lastflags = np.concatenate(lastflag_set, axis=0)
    all_ids = np.concatenate(machine_id_set, axis=0)

    # add samples of each feature after concatenating all sequences
    [
        inputs.update({feature_id: features_each})
        for feature_id, features_each in zip(feature_ids, all_features_each)
    ]
    # add labels after concatenating all sequences
    inputs.update({"label": all_labels})

    result_df["seq_id"] = all_ids
    result_df["last_flag"] = all_lastflags
    result_df["actual"] = all_labels


    # # add # of class info
    # n_labels :# of class
    inputs.update({"n_labels": len(np.unique(inputs["label"]))})

    if verbose:
        print("TOTAL SAMPLE SIZE")
        print("=================")
        print("* each feature")
        for feature_id in feature_ids:
            print(f"\t{feature_id}: {inputs[feature_id].shape}")
        print("* label", all_labels.shape)
    return inputs, result_df



def data_split(data_list,split_ratio,seed=317):
    np.random.seed(seed)
    data_len= len(data_list)
    data_ind = np.array(range(data_len))
    split_len = int(data_len*split_ratio)
    split_ind = np.random.choice(data_ind,size=split_len,replace=False)

    new_data_list = [data_list[i] for i in data_ind if not i in split_ind]
    split_list = [data_list[i] for i in data_ind if  i in split_ind]
    
    return new_data_list, split_list


def save_as_json(contains, fn, indent=4, sort_keys=True):
    with open(fn, "w") as jsonfile:
        json.dump(contains, jsonfile, indent=indent, sort_keys=sort_keys)
