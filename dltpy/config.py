import os

PROJECT_ROOT = os.path.dirname(__file__)
OUTPUT_DIRECTORY_TOPLEVEL = os.path.join('../output')
DATA_FILES_DIR = "output/"

# DLTPy paths #############################################################################
VECTOR_OUTPUT_DIRECTORY = 'output/vectors'
ML_INPUTS_PATH = "output/ml_inputs/"

ML_RETURN_DF_PATH = os.path.join(ML_INPUTS_PATH, "_ml_return.csv")
ML_PARAM_DF_PATH = os.path.join(ML_INPUTS_PATH, "_ml_param.csv")
LABEL_ENCODER_PATH = os.path.join(ML_INPUTS_PATH, "label_encoder.pkl")
DATA_FILE = os.path.join(ML_INPUTS_PATH, "_all_data.csv")
FILTERED_DATA_FILE = os.path.join(ML_INPUTS_PATH, "_data_filtered.csv")
TYPES_FILE = os.path.join(ML_INPUTS_PATH, "_most_frequent_types.csv")

OUTPUT_EMBEDDINGS_DIRECTORY = './output/resources'
W2V_MODEL_CODE_DIR = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_code_model.bin')
W2V_MODEL_LANGUAGE_DIR = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_language_model.bin')

MODEL_DIR = ".,/output/models/"
RETURN_DATAPOINTS_X = os.path.join(VECTOR_OUTPUT_DIRECTORY, "return_datapoints_x.npy")
RETURN_DATAPOINTS_Y = os.path.join(VECTOR_OUTPUT_DIRECTORY, "return_datapoints_y.npy")
PARAM_DATAPOINTS_X = os.path.join(VECTOR_OUTPUT_DIRECTORY, "param_datapoints_x.npy")
PARAM_DATAPOINTS_Y = os.path.join(VECTOR_OUTPUT_DIRECTORY, "param_datapoints_y.npy")
##########################################################################################

# TypeWriter paths #######################################################################
W2V_MODEL_ID_DIR = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_identifier_model.bin')
W2V_MODEL_TOKEN_DIR = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_token_model.bin')
W2V_MODEL_COMMENTS_DIR = os.path.join(OUTPUT_EMBEDDINGS_DIRECTORY, 'w2v_comments_model.bin')
VECTOR_OUTPUT_DIR_TW = 'output/vectors/TW'
AVAILABLE_TYPES_DIR = 'output/avl_types'
ML_INPUTS_PATH_TW = "output/ml_inputs/TW"
OUTPUT_DIRECTORY_TW = 'output/TW'

DATA_FILE_TW = os.path.join(ML_INPUTS_PATH_TW, "_all_data.csv")
#ML_RETURN_DF_PATH_TW = os.path.join(ML_INPUTS_PATH_TW, "_ml_return.csv")
#ML_PARAM_DF_PATH_TW = os.path.join(ML_INPUTS_PATH_TW, "_ml_param.csv")

ML_PARAM_TW_TRAIN = os.path.join(ML_INPUTS_PATH_TW, "_ml_param_train.csv")
ML_PARAM_TW_TEST = os.path.join(ML_INPUTS_PATH_TW, "_ml_param_test.csv")

ML_RET_TW_TRAIN = os.path.join(ML_INPUTS_PATH_TW, "_ml_ret_train.csv")
ML_RET_TW_TEST = os.path.join(ML_INPUTS_PATH_TW, "_ml_ret_test.csv")

VECTOR_OUTPUT_TRAIN = os.path.join(VECTOR_OUTPUT_DIR_TW, "train")
VECTOR_OUTPUT_TEST = os.path.join(VECTOR_OUTPUT_DIR_TW, "test")

LABEL_ENCODER_PATH_TW = os.path.join(ML_INPUTS_PATH_TW, "label_encoder.pkl")
TYPES_FILE_TW = os.path.join(ML_INPUTS_PATH_TW, "_most_frequent_types.csv")
############################################################################################

# Parameters
W2V_VEC_LENGTH = 100  # Default value of W2V
NUMBER_OF_TYPES = 1000
AVAILABLE_TYPES_NUMBER = 1000