"""
This module contains utility functions for pre-processing
"""

from os.path import join, isfile, isdir
from os import listdir, mkdir
from ast import literal_eval
from collections import Counter
import pandas as pd
import numpy as np


def filter_functions(df: pd.DataFrame, funcs=['str', 'unicode', 'repr', 'len', 'doc', 'sizeof']) -> pd.DataFrame:
    """
    Filters functions which are not useful.
    :param df: dataframe to use
    :return: filtered dataframe
    """

    df_len = len(df)
    print(f"Functions before dropping on __*__ methods {len(df)}")
    df = df[~df['name'].isin(funcs)]
    print(f"Functions after dropping on __*__ methods {len(df)}")
    print(f"Filtered out {df_len - len(df)} functions.")

    return df

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
            #if arg_name != 'self':
            arg_type = literal_eval(row['arg_types'])[p_i].strip('\"')

            # Ignore Any or None types
            # if arg_type == '' or arg_type == 'Any' or arg_type == 'None':
            #     continue
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


def filter_return_dp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters return datapoints based on a set of criteria.
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

    return df


def gen_most_frequent_avl_types(avl_types_dir, output_dir, top_n: int = 1000, save_on_disk=False):
    """
    It generates top n most frequent available types
    :param top_n:
    :return:
    """

    aval_types_files = [join(avl_types_dir, f) for f in listdir(avl_types_dir) if isfile(join(avl_types_dir, f))]

    # All available types across all Python projects
    all_aval_types = []

    for f in aval_types_files:
        with open(f, 'r') as f_aval_type:
            all_aval_types = all_aval_types + f_aval_type.read().splitlines()

    counter = Counter(all_aval_types)

    df = pd.DataFrame.from_records(counter.most_common(top_n), columns=['Types', 'Count'])

    if save_on_disk:
        df.to_csv(join(output_dir, "top_%d_types.csv", index=False))

    return df


def encode_aval_types_TW(df_param: pd.DataFrame, df_ret: pd.DataFrame, df_aval_types: pd.DataFrame):
    """
    It encodes the type of parameters and return according to the available types
    :param df_param:
    :param df_ret:
    :return:
    """

    types = df_aval_types['Types'].tolist()

    def trans_aval_type(x):
        for i, t in enumerate(types):
            if x in t:
                return i
        return len(types)

    # If the arg type doesn't exist in top_n available types, we insert n + 1 into the vector as it represents the other type.
    df_param['param_aval_enc'] = df_param['arg_type'].apply(trans_aval_type)
    df_ret['ret_aval_enc'] = df_ret['return_type'].apply(trans_aval_type)

    return df_param, df_ret


