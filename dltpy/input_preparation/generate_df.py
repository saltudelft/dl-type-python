import os
import re
from typing import Tuple
from os.path import join, isfile
from collections import Counter
import pandas as pd
import numpy as np
import pickle

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from ast import literal_eval

# LOCAL CONFIG
from dltpy import config

if not os.path.exists(config.ML_INPUTS_PATH):
    os.makedirs(config.ML_INPUTS_PATH)

# Create a path for TypeWriter
if not os.path.isdir(config.ML_INPUTS_PATH_TW):
    os.mkdir(config.ML_INPUTS_PATH_TW)

CACHE = True

def list_files(directory: str, full=True) -> list:
    """
    List all files in the given directory.

    :param directory: directory to search
    :param full: whether to return the full path
    :return: list of paths
    """
    paths = []
    for file in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, file)):
            continue

        paths.append(os.path.join(directory, file) if full else file)

    return paths


def parse_df(files: list, batch_size=10) -> pd.DataFrame:
    """
    Given a set of files, load each as a dataframe and concatenate the result
    into a large file dataframe containing the data from all files.

    :param files: the files to load
    :param batch_size: batch size used for concatenation
    :return: final dataframe
    """
    df_merged = None
    batch = []
    total = 0
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)} files")

        _loaded = pd.read_csv(file)
        total += len(_loaded)
        if df_merged is None:
            df_merged = _loaded
        else:
            batch.append(_loaded)

        if i % batch_size == 0 and i > 0:
            frames = [df_merged, *batch]
            df_merged = pd.concat(frames, ignore_index=True)
            batch = []

    if len(batch) > 0:
        df_merged = pd.concat([df_merged, *batch])

    df_merged = df_merged.reset_index(drop=True)
    assert total == len(df_merged), "Length of all loaded should be equal to merged"

    return df_merged


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the loaded dataframe.
    :type df: dataframe to use
    :returns: final dataframe
    """
    df['arg_names'] = df['arg_names'].apply(lambda x: literal_eval(x))
    df['arg_types'] = df['arg_types'].apply(lambda x: literal_eval(x))
    df['arg_descrs'] = df['arg_descrs'].apply(lambda x: literal_eval(x))
    df['return_expr'] = df['return_expr'].apply(lambda x: literal_eval(x))

    return df


def filter_functions(df: pd.DataFrame, funcs=['str', 'unicode', 'repr', 'len', 'doc', 'sizeof']) -> pd.DataFrame:
    """
    Filters functions which are not useful.
    :param df: dataframe to use
    :return: filtered dataframe
    """

    print(f"Functions before dropping on __*__ methods {len(df)}")
    df = df[~df['name'].isin(funcs)]
    print(f"Functions after dropping on __*__ methods {len(df)}")

    return df


def filter_return_dp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters return datapoints based on a set of criterias.
    """

    print(f"Functions before dropping on return type {len(df)}")
    df = df.dropna(subset=['return_type'])
    print(f"Functions after dropping on return type {len(df)}")

    print(f"Functions before dropping nan, None, Any return type {len(df)}")
    to_drop = np.invert((df['return_type'] == 'nan') | (df['return_type'] == 'None') | (df['return_type'] == 'Any'))
    df = df[to_drop]
    print(f"Functions after dropping nan return type {len(df)}")

    # Ignores return datapoints without docstrings.
    # print(f"Functions before dropping on empty docstring, function comment and return comment {len(df)}")
    # df = df.dropna(subset=['docstring', 'func_descr', 'return_descr'])
    # print(f"Functions after dropping on empty docstring, function comment and return comment {len(df)}")

    print(f"Functions before dropping on empty return expression {len(df)}")
    df = df[df['return_expr'].apply(lambda x: len(literal_eval(x))) > 0]
    print(f"Functions after dropping on empty return expression {len(df)}")

    df['return_type'] = df['return_type'].apply(lambda x: x.strip('\"'))

    return df


def gen_argument_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new dataframe containing all argument data.
    :param df: dataframe for which to extract argument
    :return: argument dataframe
    """
    arguments = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(float(i)/len(df))
        for p_i, arg_name in enumerate(literal_eval(row['arg_names'])):

            # Ignore self arg
            if arg_name != 'self':
                arg_type = literal_eval(row['arg_types'])[p_i]

                # Ignore Any or None types
                if arg_type == '' or arg_type == 'Any' or arg_type == 'None':
                    continue
                arg_descr = literal_eval(row['arg_descrs'])[p_i]

                # Ignores parameters without docstrings
                # if arg_descr == '':
                #     continue

                arguments.append([row['name'], arg_name, arg_type, arg_descr])

    return pd.DataFrame(arguments, columns=['func_name', 'arg_name', 'arg_type', 'arg_comment'])

# TypeWriter ################################################################################################

def gen_argument_df_TW(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new dataframe containing all argument data. (For Type Writer)
    :param df: dataframe for which to extract argument
    :return: argument dataframe
    """
    arguments = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(float(i)/len(df))
        for p_i, arg_name in enumerate(literal_eval(row['arg_names'])):

            # Ignore self arg
            if arg_name != 'self':
                arg_type = literal_eval(row['arg_types'])[p_i]

                # Ignore Any or None types
                if arg_type == '' or arg_type == 'Any' or arg_type == 'None':
                    continue
                arg_descr = literal_eval(row['arg_descrs'])[p_i]
                #print(arg_descr)
                #print(literal_eval(row['args_occur']))
                #print(literal_eval(row['args_occur'])[p_i])
                arg_occur = [a.replace('self', '').strip() if 'self' in a.split() else a for a in literal_eval(row['args_occur'])]

                other_args = " ".join([a for a in literal_eval(row['arg_names']) if a != 'self'])

                # Ignores parameters without docstrings
                # if arg_descr == '':
                #     continue

                arguments.append([row['file'], row['name'], row['func_descr'], arg_name, arg_type, arg_descr, other_args, arg_occur])

    return pd.DataFrame(arguments, columns=['file', 'func_name', 'func_descr', 'arg_name', 'arg_type', 'arg_comment', 'other_args',
                                            'arg_occur'])


def gen_most_frequent_avl_types(top_n: int = 1000, save_on_disk=False):
    """
    It generates top n most frequent available types
    :param top_n:
    :return:
    """

    aval_types_files = [join(config.AVAILABLE_TYPES_DIR, f) for f in os.listdir(config.AVAILABLE_TYPES_DIR) if isfile(join(config.AVAILABLE_TYPES_DIR, f))]

    # All available types across all Python projects
    all_aval_types = []

    for f in aval_types_files:

        with open(f, 'r') as f_aval_type:
            all_aval_types = all_aval_types + f_aval_type.read().splitlines()

    counter = Counter(all_aval_types)

    df = pd.DataFrame.from_records(counter.most_common(top_n), columns=['Types', 'Count'])

    if save_on_disk:
        df.to_csv(join(config.AVAILABLE_TYPES_DIR, "top_%d_types.csv" % top_n), index=False)

    return df


def encode_aval_types_TW(df_param: pd.DataFrame, df_ret: pd.DataFrame, df_aval_types: pd.DataFrame):
    """
    It encodes the type of parameters and return according to the available types
    :param df_param:
    :param df_ret:
    :return:
    """

    types = df_aval_types['Types'].tolist()

    # If the arg type doesn't exist in top_n available types, we insert n + 1 into the vector as it represents the other type.
    df_param['param_aval_enc'] = df_param['arg_type'].apply(lambda x: types.index(x) if x in types else len(types))
    df_ret['ret_aval_enc'] = df_ret['return_type'].apply(lambda x: types.index(x) if x in types else len(types))

    return df_param, df_ret

##############################################################################################################

def encode_types(df: pd.DataFrame, df_args: pd.DataFrame, enc_type_path, threshold: int = 999) -> Tuple[
    DataFrame, DataFrame, LabelEncoder]:
    """
    Encode the dataframe types to integers.
    :param df: dataframe with function data
    :param df_args: dataframe with argument data
    :param threshold: number of common types to keep
    :return: dataframe with types encoded and the labels encoder used for it.
    """
    le = preprocessing.LabelEncoder()

    # All types
    return_types = df['return_type'].values
    arg_types = df_args['arg_type'].values
    all_types = np.concatenate((return_types, arg_types), axis=0)

    unique, counts = np.unique(all_types, return_counts=True)
    print(f"Found {len(unique)} unique types in a total of {len(all_types)} types.")

    # keep the threshold most common types rest is mapped to 'other'
    common_types = [unique[i] for i in np.argsort(counts)[::-1][:threshold]]
    common_types_counts = [counts[i] for i in np.argsort(counts)[::-1][:threshold]]

    print("Remapping uncommon types for functions")
    df['return_type_t'] = df['return_type'].apply(lambda x: x if x in common_types else 'other')

    print("Remapping uncommon types for arguments")
    df_args['arg_type_t'] = df_args['arg_type'].apply(lambda x: x if x in common_types else 'other')

    print("Fitting label encoder on transformed types")
    # All types transformed
    return_types = df['return_type_t'].values
    arg_types = df_args['arg_type_t'].values
    all_types = np.concatenate((return_types, arg_types), axis=0)
    le.fit(all_types)

    print("Store type mapping with counts")
    pd.DataFrame(
        list(zip(le.transform(common_types), common_types, common_types_counts)),
        columns=['enc', 'type', 'count']
    ).to_csv(enc_type_path, index=False)

    # transform all type
    print("Transforming return types")
    df['return_type_enc'] = le.transform(return_types)

    print("Transforming args types")
    df_args['arg_type_enc'] = le.transform(arg_types)

    return df, df_args, le, unique


if __name__ == '__main__':
    if not os.path.exists(config.ML_INPUTS_PATH):
        os.makedirs(config.ML_INPUTS_PATH)

    if CACHE and os.path.exists(config.DATA_FILE):
        print("Loading cached copy")
        df = pd.read_csv(config.DATA_FILE)
    else:
        DATA_FILES = list_files(config.DATA_FILES_DIR)
        print("Found %d datafiles" % len(DATA_FILES))

        df = parse_df(DATA_FILES, batch_size=128)

        print("Dataframe loaded writing it to CSV")
        df.to_csv(config.DATA_FILE, index=False)

    # Split df
    print("Extracting arguments")
    df_params = gen_argument_df(df)

    # Filter return datapoints
    print("Filter return datapoints")
    df = filter_functions(df)

    # Format dataframe
    print("Formatting return datapoints dataframe")
    df = format_df(df)

    # Encode types as int
    print("Encoding types")
    df, df_params, label_encoder = encode_types(df, df_params)

    print("Storing label encoder")
    with open(config.LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)

    # Add argument names as a string except self
    df['arg_names_str'] = df['arg_names'].apply(lambda l: " ".join([v for v in l if v != 'self']))

    # Add return expressions as a string, replace self. and self within expressions
    df['return_expr_str'] = df['return_expr'].apply(lambda l: " ".join([re.sub(r"self\.?", '', v) for v in l]))

    # Drop all columns useless for the ML algorithms
    df = df.drop(columns=['file', 'author', 'repo', 'has_type', 'arg_names', 'arg_types', 'arg_descrs', 'return_expr'])

    # Store the dataframes
    df.to_csv(config.ML_RETURN_DF_PATH, index=False)
    df_params.to_csv(config.ML_PARAM_DF_PATH, index=False)
