"""
Author: Amir M. Mir (TU Delft)

This is the main script for using TypeWriter approach which does the following steps:
- Selects Python projects
- Extracts functions from Python repositories
- Processes the extracted functions
- Trains the embeddings
- Generates sequence vectors
"""


from os.path import join, exists, isdir
from sklearn.model_selection import train_test_split
from gh_query import load_json, gen_json_file, find_current_repos
from dltpy.input_preparation.generate_df import filter_return_dp, format_df, encode_types, list_files, parse_df
from dltpy.preprocessing.pipeline import Pipeline
from dltpy.input_preparation.df_to_vec import generate_labels
from typewriter.prepocessing import filter_functions, gen_argument_df_TW, gen_most_frequent_avl_types, \
    encode_aval_types_TW
from typewriter.extraction import IdentifierSequence, TokenSequence, CommentSequence, process_datapoints_TW, \
    gen_aval_types_datapoints
from typewriter.extraction import EmbeddingTypeWriter
from typewriter.config_TW import create_dirs, AVAILABLE_TYPES_NUMBER
from gensim.models import Word2Vec
from ast import literal_eval
from collections import Counter
import argparse
import os
import pandas as pd
import numpy as np
import re
import pickle
import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="A script for using TypeWriter approach")
    parser.add_argument("--o", required=True, type=str, help="The name of an output folder which will be created automatically")
    parser.add_argument("--d", required=True, type=str, help="Path to the Python repositories")
    parser.add_argument("--c", required=False, default=1, type=int, help="Use cached processed files if available. Use 1 for cache, otherwise 0")
    parser.add_argument("--w", required=False, default=4, type=int, help="Number of workers for extracting functions")
    parser.add_argument("--t", required=False, default=None, type=str, help="Path to the file of the visible types")
    args = parser.parse_args()

    CACHE_TW = True if args.c == 1 else False

    # Paths ###########################################################################################################
    OUTPUT_DIR = args.o
    DATASET_DIR = args.d

    SELECTED_PROJECTS_DIR = './data/selected_py_projects.json'
    #OUTPUT_EMBEDDINGS_DIRECTORY = join(OUTPUT_DIR, 'embed')
    OUTPUT_DIRECTORY_TW = join(OUTPUT_DIR, 'funcs')
    AVAILABLE_TYPES_DIR = join(OUTPUT_DIR, 'avl_types')
    RESULTS_DIR = join(OUTPUT_DIR, "results")
    TW_MODEL_FILES = join(OUTPUT_DIR, "tw_model_files")

    ML_INPUTS_PATH_TW = join(OUTPUT_DIR, "ml_inputs")
    ML_RETURN_DF_PATH_TW = join(ML_INPUTS_PATH_TW, "_ml_return.csv")
    ML_PARAM_DF_PATH_TW = join(ML_INPUTS_PATH_TW, "_ml_param.csv")
    ML_PARAM_TW_TRAIN = join(ML_INPUTS_PATH_TW, "_ml_param_train.csv")
    ML_PARAM_TW_TEST = join(ML_INPUTS_PATH_TW, "_ml_param_test.csv")
    ML_RET_TW_TRAIN = join(ML_INPUTS_PATH_TW, "_ml_ret_train.csv")
    ML_RET_TW_TEST = join(ML_INPUTS_PATH_TW, "_ml_ret_test.csv")

    VECTOR_OUTPUT_DIR_TW = join(OUTPUT_DIR, 'vectors')
    VECTOR_OUTPUT_TRAIN = join(VECTOR_OUTPUT_DIR_TW, "train")
    VECTOR_OUTPUT_TEST = join(VECTOR_OUTPUT_DIR_TW, "test")

    W2V_MODEL_TOKEN_DIR = join(TW_MODEL_FILES, 'w2v_token_model.bin')
    W2V_MODEL_COMMENTS_DIR = join(TW_MODEL_FILES, 'w2v_comments_model.bin')

    DATA_FILE_TW = join(ML_INPUTS_PATH_TW, "_all_data.csv")

    LABEL_ENCODER_PATH_TW = join(TW_MODEL_FILES, "label_encoder.pkl")
    TYPES_FILE_TW = join(ML_INPUTS_PATH_TW, "_most_frequent_types.csv")

    dirs = [OUTPUT_DIRECTORY_TW, AVAILABLE_TYPES_DIR, RESULTS_DIR, TW_MODEL_FILES, ML_INPUTS_PATH_TW,
            VECTOR_OUTPUT_DIR_TW, VECTOR_OUTPUT_TRAIN, VECTOR_OUTPUT_TEST]

    if not isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    create_dirs(dirs)
    ##################################################################################################################

    ext_time = time.time()

    # Project Selection ##############################################################################################
    # Select Python projects from the given dataset
    # Note that only projects that depends on MyPy will be selected

    if not (CACHE_TW and exists(SELECTED_PROJECTS_DIR)):
        print("Selecting Python projects.....")
        repos = load_json('./data/mypy-dependents-by-stars.json')
        gen_json_file(SELECTED_PROJECTS_DIR, repos, find_current_repos(DATASET_DIR, True))

    repos = load_json(SELECTED_PROJECTS_DIR)
    print("Number of selected Python projects:", len(repos))
    ##################################################################################################################

    # Extracting functions ###########################################################################################

    if not CACHE_TW and len(list_files(OUTPUT_DIRECTORY_TW)) == 0:
        p = Pipeline(DATASET_DIR, OUTPUT_DIRECTORY_TW, AVAILABLE_TYPES_DIR)
        p.run_pipeline_manual(repos, args.w)

    if CACHE_TW and exists(DATA_FILE_TW):
        print("Loading cached copy of the extracted functions: ", DATA_FILE_TW)
        df = pd.read_csv(DATA_FILE_TW)
    else:
        DATA_FILES = list_files(OUTPUT_DIRECTORY_TW)
        print("Found %d processed projects" % len(DATA_FILES))
        # print(DATA_FILES)
        df = parse_df(DATA_FILES, batch_size=128)
        print("Writing all the extracted functions in a CSV file: ", DATA_FILE_TW)
        df.to_csv(DATA_FILE_TW, index=False)

    print("Number of source files: ", len(df.file.unique()))
    print("Number of functions: ", len(df.name))

    print("Number of functions with comments: ",
          df[(~df['return_descr'].isnull()) | (~df['func_descr'].isnull())].shape[0])
    print("Number of functions with return types: ", df['return_type'].count())
    print("Number of functions with both: ",
          df[((~df['return_descr'].isnull()) | (~df['func_descr'].isnull())) & (~df['return_type'].isnull())].shape[0])

    # Splits the extracted functions into train and test sets by files
    train_files, test_files = train_test_split(pd.DataFrame(df['file'].unique(), columns=['file']), test_size=0.2)

    df_train = df[df['file'].isin(train_files.to_numpy().flatten())]
    df_test = df[df['file'].isin(test_files.to_numpy().flatten())]

    print("Number of functions in train set: ", df_train.shape[0])
    print("Number of functions in test set: ", df_test.shape[0])

    ##################################################################################################################

    # Processing the extracted functions #############################################################################

    df = filter_functions(df)

    # Processing parameters types
    df_params = gen_argument_df_TW(df)

    args_count = df_params['arg_name'].count()
    args_with_annot = df_params[df_params['arg_type'] != ''].shape[0]
    df_params = df_params[(df_params['arg_name'] != 'self') & ((df_params['arg_type'] != 'Any') & \
                                                               (df_params['arg_type'] != 'None'))]

    print("Number of arguments: ", args_count)
    print("Args with type annotations: ", args_with_annot)
    print("Ignored trivial types: ", (args_count - df_params.shape[0]))

    df_params = df_params[df_params['arg_type'] != '']
    print("Number of arguments with types: ", df_params.shape[0])

    # Processing return types
    df = filter_return_dp(df)
    df = format_df(df)

    df['arg_names_str'] = df['arg_names'].apply(lambda l: " ".join([v for v in l if v != 'self']))
    df['return_expr_str'] = df['return_expr'].apply(lambda l: " ".join([re.sub(r"self\.?", '', v) for v in l]))
    df = df.drop(columns=['author', 'repo', 'has_type', 'arg_names', 'arg_types', 'arg_descrs', 'return_expr'])
    
    def ext_avl_types():
        print("Extracting available type hints...")
        if CACHE_TW and exists(join(AVAILABLE_TYPES_DIR, 'top_%d_types.csv' % (AVAILABLE_TYPES_NUMBER - 1))):
            return pd.read_csv(join(AVAILABLE_TYPES_DIR, 'top_%d_types.csv' % (AVAILABLE_TYPES_NUMBER - 1)))
        else:
            return gen_most_frequent_avl_types(AVAILABLE_TYPES_DIR, TW_MODEL_FILES, AVAILABLE_TYPES_NUMBER - 1, True)

    df_types = None
    if args.t is None:
        df_types = ext_avl_types()        
    else:
        if exists(args.t):
            print("Extracting available types from an external file...")
            df_types = pd.read_csv(args.t)
            df_types = [t for type_list in df_types['types'].tolist() for t in literal_eval(type_list)]
            df_types = pd.DataFrame.from_records(Counter(df_types).most_common(AVAILABLE_TYPES_NUMBER-1),
                                                 columns=['Types', 'Count'])
            df_types.to_csv(join(TW_MODEL_FILES, "top_%d_types.csv" % (AVAILABLE_TYPES_NUMBER-1)), index=False)
        else:
            df_types = ext_avl_types()

    df_params, df = encode_aval_types_TW(df_params, df, df_types)

    # Labels of types
    df, df_params, label_encoder, uniq_types = encode_types(df, df_params, TYPES_FILE_TW)

    all_enc_types = np.concatenate((df_params['arg_type_enc'].values, df['return_type_enc'].values))
    other_type_count = np.count_nonzero(all_enc_types == label_encoder.transform(['other'])[0])
    print("Number of datapoints with other types: ", other_type_count)
    print("The percentage of covered unique types: %.2f%%" % ((AVAILABLE_TYPES_NUMBER / len(uniq_types)) * 100))
    print("The percentage of all datapoints covered by considered types: %.2f%%" % \
          ((1 - other_type_count / all_enc_types.shape[0]) * 100))

    with open(LABEL_ENCODER_PATH_TW, 'wb') as file:
        pickle.dump(label_encoder, file)

    df.to_csv(ML_RETURN_DF_PATH_TW, index=False)
    df_params.to_csv(ML_PARAM_DF_PATH_TW, index=False)

    # Splits parameters and return types into train and test
    df_params_train = df_params[df_params['file'].isin(train_files.to_numpy().flatten())]
    df_params_test = df_params[df_params['file'].isin(test_files.to_numpy().flatten())]

    df_ret_train = df[df['file'].isin(train_files.to_numpy().flatten())]
    df_ret_test = df[df['file'].isin(test_files.to_numpy().flatten())]

    print("Number of parameters types in train set: ", df_params_train.shape[0])
    print("Number of parameters types in test set: ", df_params_test.shape[0])
    print("Number of return types in train set: ", df_ret_train.shape[0])
    print("Number of return types in test set: ", df_ret_test.shape[0])

    df_params_train.to_csv(ML_PARAM_TW_TRAIN, index=False)
    df_params_test.to_csv(ML_PARAM_TW_TEST, index=False)
    df_ret_train.to_csv(ML_RET_TW_TRAIN, index=False)
    df_ret_test.to_csv(ML_RET_TW_TEST, index=False)
    ##################################################################################################################

    # Training embeddings ############################################################################################

    param_df = pd.read_csv(ML_PARAM_TW_TRAIN)
    return_df = pd.read_csv(ML_RET_TW_TRAIN)

    print("Number of parameters types:", param_df.shape[0])
    print("Number of returns types", return_df.shape[0])

    embedder = EmbeddingTypeWriter(param_df, return_df, W2V_MODEL_COMMENTS_DIR, W2V_MODEL_TOKEN_DIR)

    print("Trains embeddings for code tokens and comments....")
    embedder.train_token_model()
    embedder.train_comment_model()

    w2v_token_model = Word2Vec.load(W2V_MODEL_TOKEN_DIR)
    w2v_comments_model = Word2Vec.load(W2V_MODEL_COMMENTS_DIR)

    print("W2V statistics: ")
    print("W2V token model total amount of words : " + str(w2v_token_model.corpus_total_words))
    print("W2V comments model total amount of words : " + str(w2v_comments_model.corpus_total_words))
    print("\n Top 20 words for token model:")
    print(w2v_token_model.wv.index2entity[:20])
    print("\n Top 20 words for comments model:")
    print(w2v_comments_model.wv.index2entity[:20])
    ##################################################################################################################

    # Vector Representation ##########################################################################################

    # Parameters types
    id_trans_func_param = lambda row: IdentifierSequence(w2v_token_model, row.arg_name, row.other_args, row.func_name)
    token_trans_func_param = lambda row: TokenSequence(w2v_token_model, 7, 3, row.arg_occur, None)
    cm_trans_func_param = lambda row: CommentSequence(w2v_comments_model, row.func_descr, row.arg_comment, None)

    print("[Parameters] Generating Identifiers sequences")
    dp_ids_param_X_train = process_datapoints_TW(ML_PARAM_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'identifiers_', 'param_train',
                                                 id_trans_func_param)
    dp_ids_param_X_test = process_datapoints_TW(ML_PARAM_TW_TEST, VECTOR_OUTPUT_TEST, 'identifiers_', 'param_test',
                                                id_trans_func_param)

    print("[Parameters] Generating Tokens sequences")
    dp_tokens_param_X_train = process_datapoints_TW(ML_PARAM_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'tokens_', 'param_train',
                                                    token_trans_func_param)
    dp_tokens_param_X_test = process_datapoints_TW(ML_PARAM_TW_TEST, VECTOR_OUTPUT_TEST, 'tokens_', 'param_test',
                                                   token_trans_func_param)

    print("[Parameters] Generating Comments sequences")
    dp_cms_param_X_train = process_datapoints_TW(ML_PARAM_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'comments_', 'param_train',
                                                 cm_trans_func_param)
    dp_cms_param_X_test = process_datapoints_TW(ML_PARAM_TW_TEST, VECTOR_OUTPUT_TEST, 'comments_', 'param_test',
                                                cm_trans_func_param)

    # Returns types
    id_trans_func_ret = lambda row: IdentifierSequence(w2v_token_model, None, row.arg_names_str, row.name)
    token_trans_func_ret = lambda row: TokenSequence(w2v_token_model, 7, 3, None, row.return_expr_str)
    cm_trans_func_ret = lambda row: CommentSequence(w2v_comments_model, row.func_descr, None, row.return_descr)

    print("[Returns] Generating Identifiers sequences")
    dp_ids_ret_X_train = process_datapoints_TW(ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'identifiers_', 'ret_train',
                                               id_trans_func_ret)
    dp_ids_ret_X_test = process_datapoints_TW(ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, 'identifiers_', 'ret_test',
                                              id_trans_func_ret)

    print("[Returns] Generating Tokens sequences")
    dp_tokens_ret_X_train = process_datapoints_TW(ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'tokens_', 'ret_train',
                                                  token_trans_func_ret)
    dp_tokens_ret_X_test = process_datapoints_TW(ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, 'tokens_', 'ret_test',
                                                  token_trans_func_ret)

    print("[Returns] Generating comment sequences")
    dp_cms_ret_X_train = process_datapoints_TW(ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, 'comments_', 'ret_train',
                                               cm_trans_func_ret)
    dp_cms_ret_X_test = process_datapoints_TW(ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, 'comments_', 'ret_test',
                                              cm_trans_func_ret)

    print("[Train] Generating sequences for available types hints")
    dp_params_train_aval_types, dp_ret_train_aval_types = gen_aval_types_datapoints(ML_PARAM_TW_TRAIN,
                                                                                    ML_RET_TW_TRAIN,
                                                                                    'train',
                                                                                    VECTOR_OUTPUT_TRAIN)
    print("[Test] Generating sequences for available types hints")
    dp_params_test_aval_types, dp_ret_test_aval_types = gen_aval_types_datapoints(ML_PARAM_TW_TEST,
                                                                                  ML_RET_TW_TEST,
                                                                                  'test',
                                                                                  VECTOR_OUTPUT_TEST)

    print("[Train] Generating vector labels for types")
    params_y_train, ret_y_train = generate_labels(ML_PARAM_TW_TRAIN, ML_RET_TW_TRAIN, 'train', VECTOR_OUTPUT_TRAIN)
    print("[Test] Generating vector labels for types")
    params_y_test, ret_y_test = generate_labels(ML_PARAM_TW_TEST, ML_RET_TW_TEST, 'test', VECTOR_OUTPUT_TEST)

    ##################################################################################################################

    print("Finished the extraction of data vectors %.2f in mins" % ((time.time() - ext_time) / 60))
