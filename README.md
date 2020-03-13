# Deep Learning-based type inference for Python
This repository contains instructions for using deep learning-based approaches that predicts types for Python.

# Requirements
To install required dependencies for the approaches, run the following command:
```
pip install requirements.txt
```

# Datasets

# Approaches

## TypeWriter
The TypeWriter is proposed by:
```
Pradel, M., Gousios, G., Liu, J., & Chandra, S. (2019). TypeWriter: Neural Type Prediction with Search-based Validation. arXiv preprint arXiv:1912.03768.
```

### Script

The re-implementation of this approach consists of two parts:

 - Extractor: To process functions and generate data vectors, use the following script:
```
python TW_extractor.py --o $OUTPUT_FOLDER --d $REPOS --w $THREADS
```
`$OUTPUT_FOLDER`: Specify a folder to store the output files. The folder will be created automatically.

`$REPOS`: The path to Python projects

`$THREADS`: Number of workers for extracting functions from the projects.

- Model: To learn the neural model and do prediction, run the following script:
```
python TW_model.py --o $OUTPUT_FOLDER
```
`$OUTPUT_FOLDER`: Note that this is the output folder that you specified in the extractor part.

To change the hyper-parameters of the neural model, you can change the values in the file `data/tw_model_learning_params.json`.

### Notebook
To run all the steps of TypeWriter manually with explanation, check out the notebook `main_TW.ipynb`.

