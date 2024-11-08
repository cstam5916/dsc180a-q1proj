# DSC180-Q1 Project

This repository contains code for the **Quarter 1 Project** for the course **DSC180A: Senior Capstone Project I**, taken in the Fall of 2024 at **UC San Diego**.

## Project Structure

The repository consists of five Python scripts:
- Three scripts define classes for various **Graph Neural Network (GNN) models**.
- Two scripts provide training loops to test and compare these models on different datasets.

### Model Files
Each model file includes two versions of the respective GNN model:
1. **Node-Level Embedding Model** - For generating embeddings at the node level.
2. **Graph-Level Inference Model** - For performing inference on an entire graph.

The models are:

1. `GCN.py` - **Graph Convolutional Network**
2. `GIN.py` - **Graph Isomorphism Network**
3. `GAT.py` - **Graph Attention Network**

### Training Files
These models are tested using the following training scripts:

1. `train_cora.py` - Implements a training loop for **node classification** on the **CORA dataset**.
2. `train_enzymes.py` - Implements a training loop for **graph-level classification** on the **ENZYMES dataset**.

### To Replicate
The scripts in this project should be easy to run assuming that the user has a successful installation of PyTorch Geometric. A list of the packages I personally have installed to the Anaconda environment I use for this project can be found in packagelist.txt. From there, the user can simply run the python files `train_cora.py` and `train_enzymes.py`.
