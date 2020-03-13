from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

from pandas import Series

from dltpy import config

# Create a path for DLTPy
if not os.path.isdir(config.VECTOR_OUTPUT_DIRECTORY):
    os.mkdir(config.VECTOR_OUTPUT_DIRECTORY)

# Create paths for TypeWriter #####################
if not os.path.isdir(config.VECTOR_OUTPUT_DIR_TW):
    os.mkdir(config.VECTOR_OUTPUT_DIR_TW)

if not os.path.isdir(config.VECTOR_OUTPUT_TRAIN):
    os.mkdir(config.VECTOR_OUTPUT_TRAIN)

if not os.path.isdir(config.VECTOR_OUTPUT_TEST):
    os.mkdir(config.VECTOR_OUTPUT_TEST)
###################################################


def vectorize_string(sentence: str, feature_length: int, w2v_model: Word2Vec) -> np.ndarray:
    """
    Vectorize a sentence to a multi-dimensial NumPy array

    Roughly based on https://github.com/sola-da/NL2Type/blob/master/scripts/csv_to_vecs.py
    """

    vector = np.zeros((feature_length, config.W2V_VEC_LENGTH))

    for i, word in enumerate(sentence.split()):
        if i >= feature_length:
            break
        try:
            vector[i] = w2v_model.wv[word]
        except KeyError:
            pass

    return vector


class Datapoint(ABC):
    """
    Abstract class to represent a datapoint
    """

    @property
    @abstractmethod
    def feature_lengths(self) -> Dict[str, int]:
        """
        The lengths (number of vectors) of the features
        """
        pass

    @property
    @abstractmethod
    def feature_types(self) -> Dict[str, str]:
        """
        The types (datapoint_type, code or language) of the features
        """
        pass

    def vector_length(self) -> int:
        """
        The length of the whole vector for this datapoint
        """
        return sum(self.feature_lengths.values()) + len(self.feature_lengths.values()) - 1

    def to_vec(self) -> np.ndarray:
        """
        The vector for this datapoint

        The vector contains all the features specified in the subclass. Natural language features are converted to
        vectors using the word2vec models.
        """

        w2v_models = {
            'code': Word2Vec.load(config.W2V_MODEL_CODE_DIR),
            'language': Word2Vec.load(config.W2V_MODEL_LANGUAGE_DIR)
        }

        datapoint = np.zeros((self.vector_length(), config.W2V_VEC_LENGTH))

        separator = np.ones(config.W2V_VEC_LENGTH)

        position = 0
        for feature, feature_length in self.feature_lengths.items():

            if self.feature_types[feature] == 'datapoint_type':
                datapoint[position] = self.datapoint_type_vector()
                position += 1

            if self.feature_types[feature] == 'code' or self.feature_types[feature] == 'language':
                vectorized_feature = vectorize_string(
                    self.__dict__[feature] if isinstance(self.__dict__[feature], str) else '',
                    feature_length,
                    w2v_models[self.feature_types[feature]]
                )

                for word in vectorized_feature:
                    datapoint[position] = word
                    position += 1

            if self.feature_types[feature] == 'padding':
                for i in range(0, feature_length):
                    datapoint[position] = np.zeros(config.W2V_VEC_LENGTH)
                    position += 1

            # Add separator after each feature
            if position < len(datapoint):
                datapoint[position] = separator
                position += 1

        return datapoint

    def to_be_predicted_to_vec(self) -> np.ndarray:
        """
        A vector representation of what needs to be predicted, in this case the type
        """
        vector = np.zeros(config.NUMBER_OF_TYPES)
        vector[self.type] = 1
        return vector

    @abstractmethod
    def datapoint_type_vector(self) -> np.ndarray:
        """
        The vector corresponding to the type
        """
        pass

    def __repr__(self) -> str:
        values = list(map(lambda kv: kv[0] + ': ' + repr(kv[1]), self.__dict__.items()))
        values = "\n\t" + ",\n\t".join(values) + "\n"
        return type(self).__name__ + "(%s)" % values


class ParameterDatapoint(Datapoint):
    """
    A parameter data point representing the tuple (n_p_i, c_p_i)
    """
    @property
    def feature_lengths(self) -> Dict[str, int]:
        return {
            'datapoint_type': 1,
            'name': 6,
            'comment': 15,
            'padding_0': 6,
            'padding_1': 12,
            'padding_2': 10
        }

    @property
    def feature_types(self) -> Dict[str, str]:
        return {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'comment': 'language',
            'padding_0': 'padding',
            'padding_1': 'padding',
            'padding_2': 'padding'
        }

    def __init__(self, name: str, comment: str, type: int):
        self.name = name
        self.comment = comment
        self.type = type

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, config.W2V_VEC_LENGTH))
        datapoint_type[0][0] = 1
        return datapoint_type


class ReturnDatapoint(Datapoint):
    """
    A return data point representing the tuple (n_f, c_f, r_c, r_e, n_p)
    """
    @property
    def feature_lengths(self) -> Dict[str, int]:
        return {
            'datapoint_type': 1,
            'name': 6,
            'function_comment': 15,
            'return_comment': 6,
            'return_expressions': 12,
            'parameter_names': 10
        }

    @property
    def feature_types(self) -> Dict[str, str]:
        return {
            'datapoint_type': 'datapoint_type',
            'name': 'code',
            'function_comment': 'language',
            'return_comment': 'language',
            'return_expressions': 'code',
            'parameter_names': 'code'
        }

    def __init__(self, name: str, function_comment: str, return_comment: str, return_expressions: list,
                 parameter_names: list, type: int):
        self.name = name
        self.function_comment = function_comment
        self.return_comment = return_comment
        self.return_expressions = return_expressions
        self.parameter_names = parameter_names
        self.type = type

    def datapoint_type_vector(self) -> np.ndarray:
        datapoint_type = np.zeros((1, config.W2V_VEC_LENGTH))
        datapoint_type[0][1] = 1
        return datapoint_type


def process_datapoints(filename: str, type: str, transformation: Callable[[Series], Datapoint]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Read dataframe, generate vectors for each row, and write them as multidimensional array to disk
    """
    print(f'Generating input vectors for {type} datapoints')

    df = pd.read_csv(filename)

    datapoints = df.apply(transformation, axis=1)

    datapoints_result_x = np.stack(datapoints.apply(lambda x: x.to_vec()), axis=0)
    np.save(os.path.join(config.VECTOR_OUTPUT_DIRECTORY, type + '_datapoints_x'), datapoints_result_x)
    datapoints_result_y = np.stack(datapoints.apply(lambda x: x.to_be_predicted_to_vec()), axis=0)
    np.save(os.path.join(config.VECTOR_OUTPUT_DIRECTORY, type + '_datapoints_y'), datapoints_result_y)

    return datapoints_result_x, datapoints_result_y

# TypeWriter ################################################################################################


class IdentifierEmbedding:

    def __init__(self, identifiers_embd, arg_name, args_name, func_name):

        self.identifiers_embd = identifiers_embd
        self.arg_name = arg_name
        self.args_name = args_name
        self.func_name = func_name

    def seq_length_param(self):

        return {
            "arg_name": 10,
            "sep": 1,
            "func_name": 10,
            "args_name": 10
        }

    def seq_length_return(self):

        return {
            "func_name": 10,
            "sep": 1,
            "args_name": 10,
            "padding": 10
        }

    def generate_datapoint(self, seq_length):

        datapoint = np.zeros((sum(seq_length.values()), config.W2V_VEC_LENGTH))
        separator = np.ones(config.W2V_VEC_LENGTH)

        p = 0
        for seq, length in seq_length.items():

            if seq == "sep":

                datapoint[p] = separator
                p += 1

            elif seq == 'padding':

                for i in range(0, length):
                    datapoint[p] = np.zeros(config.W2V_VEC_LENGTH)
                    p += 1
            else:

                try:

                    for w in vectorize_string(self.__dict__[seq], length, self.identifiers_embd):
                        datapoint[p] = w
                        p += 1
                # TODO: There are NaN values for func and arg names...
                except AttributeError:
                    #print(self.__dict__)
                    pass

        return datapoint

    def param_datapoint(self):

        return self.generate_datapoint(self.seq_length_param())

    def return_datapoint(self):

        return self.generate_datapoint(self.seq_length_return())


class TokenEmbedding:

    def __init__(self, token_model, len_tk_seq, num_tokens_seq, args_usage, return_expr):
        self.token_model = token_model
        self.len_tk_seq = len_tk_seq
        self.num_tokens_seq = num_tokens_seq
        self.args_usage = args_usage
        self.return_expr = return_expr

    def param_datapoint(self):

        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, config.W2V_VEC_LENGTH))

        p = 0
        for i, u in enumerate(self.args_usage):
            if i >= self.num_tokens_seq:
                break
            for w in vectorize_string(u, self.len_tk_seq, self.token_model):
                datapoint[p] = w
                p += 1

        return datapoint

    def return_datapoint(self):

        # TODO: Later consider all the return expressions
        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, config.W2V_VEC_LENGTH))

        if isinstance(self.return_expr, str):
            p = 0
            for w in vectorize_string(self.return_expr, self.len_tk_seq, self.token_model):
                datapoint[p] = w
                p += 1

        return datapoint

        # try:
        #
        #     return vectorize_string(self.return_expr, self.len_tk_seq, self.token_model)
        # # TODO: some return expressions cannot be processed
        # except AttributeError:
        #
        #     pass


class CommentEmbedding:

    def __init__(self, cm_model, func_cm, args_cm, ret_cm):
        self.cm_model = cm_model
        self.func_cm = func_cm
        self.args_cm = args_cm
        self.ret_cm = ret_cm

    # def seq_length_param(self):

    #     return {
    #         "args_cm": 20,
    #         "padding": 1
    #     }

    # def seq_length_return(self):

    #     return {
    #         "func_cm": 10,
    #         "sep": 1,
    #         "args_cm": 10
    #     }

    def seq_length_param(self):

         return {
            "func_cm": 20,
            "args_cm": 20
            }

    def seq_length_return(self):

        return {
            "func_cm": 20,
            "ret_cm": 20
            }

    def generate_datapoint(self, seq_length):

        datapoint = np.zeros((sum(seq_length.values()), config.W2V_VEC_LENGTH))
        separator = np.ones(config.W2V_VEC_LENGTH)

        p = 0
        for seq, length in seq_length.items():

            if seq == "sep":

                datapoint[p] = separator
                p += 1

            elif seq == 'padding':

                for i in range(0, length):
                    datapoint[p] = np.zeros(config.W2V_VEC_LENGTH)
                    p += 1

            else:

                try:

                    for w in vectorize_string(self.__dict__[seq], length, self.cm_model):
                        datapoint[p] = w
                        p += 1
                # TODO: There are NaN values for func and arg names...
                except AttributeError:
                    #print(self.__dict__)
                    pass

        return datapoint

    def param_datapoint(self):

        return self.generate_datapoint(self.seq_length_param())

    def return_datapoint(self):

        return self.generate_datapoint(self.seq_length_return())


def process_datapoints_TW(f_name, output_path, embedding_type, type, trans_func):

    df = pd.read_csv(f_name)
    datapoints = df.apply(trans_func, axis=1)

    datapoints_X = np.stack(datapoints.apply(lambda x: x.return_datapoint() if 'ret' in type else x.param_datapoint()),
                            axis=0)
    np.save(os.path.join(output_path, embedding_type + type + '_datapoints_x'), datapoints_X)

    return datapoints_X


def type_vector(size, index):
    v = np.zeros(size)
    v[index] = 1
    return v


def gen_aval_types_datapoints(df_params, df_ret, set_type, output_path):
    """
    It generates data points for available types.
    :param df_aval_types:
    :return:
    """

    df_params = pd.read_csv(df_params)
    df_ret = pd.read_csv(df_ret)

    aval_types_params = np.stack(df_params.apply(lambda row: type_vector(config.AVAILABLE_TYPES_NUMBER, row.param_aval_enc),
                                                 axis=1), axis=0)
    aval_types_ret = np.stack(df_ret.apply(lambda row: type_vector(config.AVAILABLE_TYPES_NUMBER, row.ret_aval_enc),
                                           axis=1), axis=0)

    np.save(os.path.join(output_path, f'params_{set_type}_aval_types_dp'), aval_types_params)
    np.save(os.path.join(output_path, f'ret_{set_type}_aval_types_dp'), aval_types_ret)

    return aval_types_params, aval_types_ret


def generate_labels(params_df, returns_df, set_type, output_path):

    params_df = pd.read_csv(params_df)
    returns_df = pd.read_csv(returns_df)

    params_y = np.stack(params_df.apply(lambda row: type_vector(config.NUMBER_OF_TYPES, row.arg_type_enc), axis=1),
                        axis=0)
    returns_y = np.stack(returns_df.apply(lambda row: type_vector(config.NUMBER_OF_TYPES, row.return_type_enc), axis=1),
                        axis=0)

    np.save(os.path.join(output_path, f'params_{set_type}_datapoints_y'), params_y)
    np.save(os.path.join(output_path, f'ret_{set_type}_datapoints_y'), returns_y)

    return params_y, returns_y

#############################################################################################################





if __name__ == '__main__':
    if not os.path.isdir(config.VECTOR_OUTPUT_DIRECTORY):
        os.mkdir(config.VECTOR_OUTPUT_DIRECTORY)

    # Process parameter datapoints
    param_datapoints_result_x, param_datapoints_result_y = process_datapoints(
        config.ML_PARAM_DF_PATH,
        'param',
        lambda row: ParameterDatapoint(row.arg_name, row.arg_comment, row.arg_type_enc)
    )

    return_datapoints_result_x, return_datapoints_result_y = process_datapoints(
        config.ML_RETURN_DF_PATH,
        'return',
        lambda row: ReturnDatapoint(row['name'], row.func_descr if row.func_descr is str else row.docstring,
                                    row.return_descr, row.return_expr_str, row.arg_names_str, row.return_type_enc),
    )

    assert param_datapoints_result_x.shape[1] == return_datapoints_result_x.shape[1], \
        "Param datapoints and return datapoints must have the same length, thus padding must be added."
