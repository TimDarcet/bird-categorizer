Caltech200 bird dataset classifier challenge
============================================

This is my project for the image classification challenge as part of the image recognition course at MVA.

For more information on the methods I used, please read the file `MVA_RecVis_TD3_report.pdf`


### Codebase structure
- **Cropping**: `cropper.py` is used to crop an initial dataset into a cropped, cleaner dataset
- **Training and testing**:
    - `data.py` defines the data transforms and augmentations
    - `models` is the folder containing all model files.
    - `model.py` defines the model used as `Net`. In practice, `model.py` is a symlink to a file in the folder `models`.
    - `main.py` is a simple training script with an `argparse` CLI.
    - `evaluate.py` is a simple evaluation script featuring an `argparse` CLI to evaluate on the test dataset.
- **Distributing computations**:
    - `computers_name` is the list of the adresses of computers usable for training, one on each line. It is not included here.
    - `distrib_train.py` is a more advanced distributed training script using Pytorch `torch.nn.parallel` module and `DistributedDataParallel` function. It features an `argparse` CLI, but should be launched using Pytorch's `launch.py` script. See also `launch.sh` for launching.
    - `launch.sh` is a bash script to launch a worker on a specified number of nodes, taking node adresses from `computers_name`. It spawns a tmux on each node to allow for easily attaching to its shell and easy killing, and logs to the folders `logs/stdout` and `logs/stderr`
    - `kill.sh` does the inverse of `launch.sh`, it kills workers on a specified number of nodes.
- **Visualising features**
    - `visualise_features.py` is a script to visualise the outputs of a neural network on the dataset using PCA and t-SNE. It uses the model `Net` defined in `featurizer.py`. I used it to visualise the activations of the layer just before classification, in order to understand better the inner workings of my models.
    - `featurizer.py` defines the model used for this. Usually a symlink to a model in `models`.
    - `TSNE_embeddings` is a folder containing pictures of such embeddings.
- `workshop.ipynb` is a notebook that I used to quickly try different things.
- `MVA_RecVis_TD3_report.pdf` is my formal report where I detail the methods I used.

