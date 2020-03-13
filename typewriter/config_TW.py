"""
Global config for TypeWriter
"""

from os.path import join, isdir
from os import mkdir

# Parameters
W2V_VEC_LENGTH = 100  # Default value of W2V
AVAILABLE_TYPES_NUMBER = 1000
CACHE_TW = True  # Read from pro-processed files

# Paths
# OUTPUT_DIR = './output'

# OUTPUT_EMBEDDINGS_DIRECTORY = lambda: join(OUTPUT_DIR, 'embed')
# OUTPUT_DIRECTORY_TW = lambda: join(OUTPUT_DIR, 'funcs')
# AVAILABLE_TYPES_DIR = lambda: join(OUTPUT_DIR, 'avl_types')
# RESULTS_DIR = lambda: join(OUTPUT_DIR, "results")

# ML_INPUTS_PATH_TW = lambda: join(OUTPUT_DIR, "ml_inputs")
# ML_PARAM_TW_TRAIN = lambda: join(ML_INPUTS_PATH_TW(), "_ml_param_train.csv")
# ML_PARAM_TW_TEST = lambda: join(ML_INPUTS_PATH_TW(), "_ml_param_test.csv")
# ML_RET_TW_TRAIN = lambda: join(ML_INPUTS_PATH_TW(), "_ml_ret_train.csv")
# ML_RET_TW_TEST = lambda: join(ML_INPUTS_PATH_TW(), "_ml_ret_test.csv")

# VECTOR_OUTPUT_DIR_TW = lambda: join(OUTPUT_DIR, 'vectors')
# VECTOR_OUTPUT_TRAIN = lambda: join(VECTOR_OUTPUT_DIR_TW(), "train")
# VECTOR_OUTPUT_TEST = lambda: join(VECTOR_OUTPUT_DIR_TW(), "test")

# W2V_MODEL_TOKEN_DIR = lambda: join(OUTPUT_EMBEDDINGS_DIRECTORY(), 'w2v_token_model.bin')
# W2V_MODEL_COMMENTS_DIR = lambda: join(OUTPUT_EMBEDDINGS_DIRECTORY(), 'w2v_comments_model.bin')

# DATA_FILE_TW = lambda: join(ML_INPUTS_PATH_TW(), "_all_data.csv")

# LABEL_ENCODER_PATH_TW = lambda: join(ML_INPUTS_PATH_TW(), "label_encoder.pkl")
# TYPES_FILE_TW = lambda: join(ML_INPUTS_PATH_TW(), "_most_frequent_types.csv")


def create_dirs(dirs):
    """
    Creates all the required dirs required by the project
    :return:
    """

    # dirs = [OUTPUT_EMBEDDINGS_DIRECTORY, OUTPUT_DIRECTORY_TW, AVAILABLE_TYPES_DIR, RESULTS_DIR,
    #         ML_INPUTS_PATH_TW, VECTOR_OUTPUT_DIR_TW, VECTOR_OUTPUT_TRAIN, VECTOR_OUTPUT_TEST]

    # if not isdir(OUTPUT_DIR):
    #     mkdir(OUTPUT_DIR)

    for d in dirs:
        if not isdir(d):
            mkdir(d)
            print("created folder ", d)
