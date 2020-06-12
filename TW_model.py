"""
Author: Amir M. Mir (TU Delft)

This is the main script for learning the neural model TypeWriter and prediction.
"""

from gh_query import load_json
from typewriter.config_TW import W2V_VEC_LENGTH, AVAILABLE_TYPES_NUMBER
from typewriter.model import load_data_tensors_TW, load_label_tensors_TW, EnhancedTWModel, train_loop_TW, evaluate_TW, \
    report_TW
from os.path import join, abspath
from torch.utils.data import DataLoader, TensorDataset
from statistics import mean
from datetime import datetime
import argparse
import torch
import pickle
import time
import result_proc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Learns the neural model of TypeWriter")
    parser.add_argument("--o", required=True, type=str, help="The path to data vectors")
    #parser.add_argument("--r", required=True, type=str, help="The path to store the results of prediction")
    #parser.add_argument("--j", required=True, type=str, help="The path to JSON file of learning parameters")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    OUTPUT_DIR = args.o
    RESULTS_DIR = join(OUTPUT_DIR, "results")
    ML_INPUTS_PATH_TW = join(OUTPUT_DIR, "ml_inputs")
    TW_MODEL_FILES = join(OUTPUT_DIR, "tw_model_files")

    VECTOR_OUTPUT_DIR_TW = join(OUTPUT_DIR, 'vectors')
    VECTOR_OUTPUT_TRAIN = join(VECTOR_OUTPUT_DIR_TW, "train")
    VECTOR_OUTPUT_TEST = join(VECTOR_OUTPUT_DIR_TW, "test")

    LABEL_ENCODER_PATH_TW = join(TW_MODEL_FILES, "label_encoder.pkl")
    #################################################################################################################

    # Helper functions for loading data vectors #######################################################################

    def load_param_train_data():
        return load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'identifiers_param_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'tokens_param_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'comments_param_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'params_train_aval_types_dp.npy')), \
               load_label_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'params_train_datapoints_y.npy'))

    def load_param_test_data():
        return load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'identifiers_param_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'tokens_param_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'comments_param_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'params_test_aval_types_dp.npy')), \
               load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, 'params_test_datapoints_y.npy'))


    def load_ret_train_data():
        return load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'identifiers_ret_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'tokens_ret_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'comments_ret_train_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'ret_train_aval_types_dp.npy')), \
               load_label_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'ret_train_datapoints_y.npy'))


    def load_ret_test_data():
        return load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'identifiers_ret_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'tokens_ret_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'comments_ret_test_datapoints_x.npy')), \
               load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'ret_test_aval_types_dp.npy')), \
               load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, 'ret_test_datapoints_y.npy'))

    def load_combined_train_data():
        return torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'identifiers_param_train_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'identifiers_ret_train_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'tokens_param_train_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'tokens_ret_train_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'comments_param_train_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'comments_ret_train_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'params_train_aval_types_dp.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'ret_train_aval_types_dp.npy')))), \
               torch.cat((load_label_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'params_train_datapoints_y.npy')),
                          load_label_tensors_TW(join(VECTOR_OUTPUT_TRAIN, 'ret_train_datapoints_y.npy'))))


    def load_combined_test_data():
        return torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'identifiers_param_test_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'identifiers_ret_test_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'tokens_param_test_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'tokens_ret_test_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'comments_param_test_datapoints_x.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'comments_ret_test_datapoints_x.npy')))), \
               torch.cat((load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'params_test_aval_types_dp.npy')),
                          load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, 'ret_test_aval_types_dp.npy')))), \
               torch.cat((load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, 'params_test_datapoints_y.npy')),
                          load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, 'ret_test_datapoints_y.npy'))))

    datasets_train = {'combined': load_combined_train_data,
                      'return': load_ret_train_data,
                      'argument': load_param_train_data}
    datasets_test = {'combined': load_combined_test_data,
                     'return': load_ret_test_data,
                     'argument': load_param_test_data}

    #################################################################################################################

    # Learning parameters ##########################################################################################
    print("Reading the learning parameters from the JSON file...")
    learn_params = load_json("./data/tw_model_learning_params.json")

    input_size = W2V_VEC_LENGTH
    hidden_size = learn_params['hidden_size']
    output_size = learn_params['output_size']
    num_layers = learn_params['num_layers']
    learning_rate = learn_params['learning_rate']
    dropout_rate = learn_params['dropout_rate']
    epochs = learn_params['epochs']
    top_n_pred = learn_params['top_n_pred']
    n_rep = learn_params['n_rep']
    batch_size = learn_params['batch_size']
    #train_split_size = 0.8
    data_loader_workers = learn_params['data_loader_workers']
    params_dict = {'epochs': epochs, 'lr': learning_rate, 'dr': dropout_rate,
                   'batches': batch_size, 'layers': num_layers, 'hidden_size': hidden_size}
    #################################################################################################################

    # Training the model ############################################################################################
    model = EnhancedTWModel(input_size, hidden_size, AVAILABLE_TYPES_NUMBER, num_layers, output_size,
                            dropout_rate).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    idx_of_other = pickle.load(open(LABEL_ENCODER_PATH_TW, 'rb')).transform(['other'])[0]
    res_time = datetime.now().strftime("%b%d_%H-%M-%S")
    for d in datasets_train:
        print(f"Loading {d} data for model {model.module.__class__.__name__}")
        # X_id, X_tok, X_cm, X_type, Y = datasets[d]
        load_data_t = time.time()
        X_id_train, X_tok_train, X_cm_train, X_type_train, Y_train = datasets_train[d]()
        X_id_test, X_tok_test, X_cm_test, X_type_test, Y_test = datasets_test[d]()

        train_loader = DataLoader(TensorDataset(X_id_train, X_tok_train, X_cm_train, X_type_train,
                                                Y_train), batch_size=batch_size, shuffle=True,
                                                pin_memory=True, num_workers=data_loader_workers)

        test_loader = DataLoader(TensorDataset(X_id_test, X_tok_test, X_cm_test, X_type_test,
                                               Y_test), batch_size=batch_size)
        print("Loaded train and test sets in %.2f min" % ((time.time() - load_data_t) / 60))

        for i in range(1, n_rep + 1):

            train_t = time.time()
            train_loop_TW(model, train_loader, learning_rate, epochs)
            print("Training finished in %.2f min" % ((time.time() - train_t) / 60))
            eval_t = time.time()
            y_true, y_pred = evaluate_TW(model, test_loader, top_n=max(top_n_pred))
            print("Prediction finished in %.2f min" % ((time.time() - eval_t) / 60))

            # Ignore other type
            idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)
            f1_score_top_n = []
            for top_n in top_n_pred:
                filename = f"{model.module.__class__.__name__ if torch.cuda.device_count() > 1 else model.__class__.__name__}_{d}_{i}_{top_n}"
                report_TW(y_true, y_pred, top_n, f"{filename}_unfiltered_{res_time}", RESULTS_DIR, params_dict)
                report = report_TW(y_true[idx], y_pred[idx], top_n, f"{filename}_filtered_{res_time}", RESULTS_DIR, params_dict)
                f1_score_top_n.append(report['result']['macro avg']['f1-score'])
            print("Mean f1_score:", mean(f1_score_top_n))
            
            # Saving the model ###############################################################################################
            torch.save(model.module if torch.cuda.device_count() > 1 else model, join(TW_MODEL_FILES, 'tw_pretrained_model_%s.pt' % d))
            print("Saved the neural model of TyperWriter at:\n%s" % abspath(join(TW_MODEL_FILES, 'tw_pretrained_model_%s.pt' % d)))
            ##################################################################################################################

            if torch.cuda.device_count() > 1:
                model.module.reset_model_parameters()
            else:
                model.reset_model_parameters()
    ##################################################################################################################

    # Prediction Results ############################################################################################
    for p in datasets_train.keys():

        res = result_proc.eval_result(RESULTS_DIR, 'EnhancedTWModel', res_time, p, 'filtered', True)
        print(f"-------------- Prediction results for {p} --------------")
        for t, r in res.items():
            print(f"{t}: F1-score: {format(r['f1-score'] * 100, '.2f')} - Recall: {format(r['recall'] * 100, '.2f')} - Precision: {format(r['precision'] * 100, '.2f')}")
    ##################################################################################################################
