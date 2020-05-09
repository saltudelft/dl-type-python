"""
This module contains embeddings and vector representations for TypeWriter
"""

from gensim.models import Word2Vec
from typewriter.config_TW import W2V_VEC_LENGTH, AVAILABLE_TYPES_NUMBER
from dltpy.input_preparation.df_to_vec import vectorize_string
from time import time
from os.path import isdir, join
from os import mkdir
import pandas as pd
import numpy as np
import multiprocessing


# Embeddings ##########################################################################################################

class HelperIterator:
    """
    Subclass for the iterators
    """
    pass


class BaseEmbedder:
    """
    Create embeddings for the code names and docstring names using Word2Vec.
    """

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def train_model(self, corpus_iterator: HelperIterator, model_path_name: str) -> None:
        """
        Train a Word2Vec model and save the output to a file.
        :param corpus_iterator: class that can provide an iterator that goes through the corpus
        :param model_path_name: path name of the output file
        """

        cores = multiprocessing.cpu_count()

        w2v_model = Word2Vec(min_count=5,
                             window=5,
                             size=W2V_VEC_LENGTH,
                             workers=cores-1)

        t = time()

        w2v_model.build_vocab(sentences=corpus_iterator)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        w2v_model.train(sentences=corpus_iterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=20,
                        report_delay=1)

        print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))

        w2v_model.save(model_path_name)


class EmbeddingTypeWriter(BaseEmbedder):

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame, w2v_model_cm_path, w2v_model_tk_path) -> None:
        super().__init__(param_df, return_df)
        self.w2v_model_cm_path = w2v_model_cm_path
        self.w2v_model_tk_path = w2v_model_tk_path

    def train_token_model(self):
        """
        Trains a W2V model for tokens.
        """

        self.train_model(TokenIterator(self.param_df, self.return_df), self.w2v_model_tk_path)

    def train_comment_model(self):
        """
        Trains a W2V model for comments.
        """

        self.train_model(CommentIterator(self.param_df, self.return_df), self.w2v_model_cm_path)


class CommentIterator(HelperIterator):

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def __iter__(self):

        for func_descr_sentence in self.return_df['func_descr'][~self.return_df['func_descr'].isnull()]:
            yield func_descr_sentence.split()

        for return_descr_sentence in self.return_df['return_descr'][~self.return_df['return_descr'].isnull()]:
            yield return_descr_sentence.split()

        for param_descr_sentence in self.param_df['arg_comment'][~self.param_df['arg_comment'].isnull()]:
            yield param_descr_sentence.split()


class TokenIterator(HelperIterator):

    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def __iter__(self):
        for return_expr_sentences in self.return_df['return_expr_str'][~self.return_df['return_expr_str'].isnull()]:
            yield return_expr_sentences.split()

        for code_occur_sentences in self.param_df['arg_occur'][~self.param_df['arg_occur'].isnull()]:
            yield code_occur_sentences.split()

        for func_name_sentences in self.param_df['func_name'][~self.param_df['func_name'].isnull()]:
            yield func_name_sentences.split()

        for arg_names_sentences in self.return_df['arg_names_str'][~self.return_df['arg_names_str'].isnull()]:
            yield arg_names_sentences.split()

######################################################################################################################

# Sequences #########################################################################################################


class IdentifierSequence:

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

        datapoint = np.zeros((sum(seq_length.values()), W2V_VEC_LENGTH))
        separator = np.ones(W2V_VEC_LENGTH)

        p = 0
        for seq, length in seq_length.items():

            if seq == "sep":

                datapoint[p] = separator
                p += 1

            elif seq == 'padding':

                for i in range(0, length):
                    datapoint[p] = np.zeros(W2V_VEC_LENGTH)
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


class TokenSequence:

    def __init__(self, token_model, len_tk_seq, num_tokens_seq, args_usage, return_expr):
        self.token_model = token_model
        self.len_tk_seq = len_tk_seq
        self.num_tokens_seq = num_tokens_seq
        self.args_usage = args_usage
        self.return_expr = return_expr

    def param_datapoint(self):

        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, W2V_VEC_LENGTH))

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
        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, W2V_VEC_LENGTH))

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


class CommentSequence:

    def __init__(self, cm_model, func_cm, args_cm, ret_cm):
        self.cm_model = cm_model
        self.func_cm = func_cm
        self.args_cm = args_cm
        self.ret_cm = ret_cm

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

        datapoint = np.zeros((sum(seq_length.values()), W2V_VEC_LENGTH))
        separator = np.ones(W2V_VEC_LENGTH)

        p = 0
        for seq, length in seq_length.items():

            if seq == "sep":

                datapoint[p] = separator
                p += 1

            elif seq == 'padding':

                for i in range(0, length):
                    datapoint[p] = np.zeros(W2V_VEC_LENGTH)
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
    np.save(join(output_path, embedding_type + type + '_datapoints_x'), datapoints_X)

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

    aval_types_params = np.stack(df_params.apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.param_aval_enc),
                                                 axis=1), axis=0)
    aval_types_ret = np.stack(df_ret.apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.ret_aval_enc),
                                           axis=1), axis=0)

    np.save(join(output_path, f'params_{set_type}_aval_types_dp'), aval_types_params)
    np.save(join(output_path, f'ret_{set_type}_aval_types_dp'), aval_types_ret)

    return aval_types_params, aval_types_ret


######################################################################################################################
