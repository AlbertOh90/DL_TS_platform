**A Deep Learning based Time Series Forecasting Platform (PyTorch version)**

This platform is designed with two considerations: 1. The architecture and training modules of neural networks
can be reused for different tasks like a Lego. 2. NN modules, losses and training function are easier exchangable in order to search best combinations for a specific task.

**TS platform**

The platform consists of four components: 
- encoders: Contains list of backbones of the NN models that encoding the input data into features before any loss functions
- trainers: List of trainers to cope with different loss functions and training data formats.
- TS_classifer&TS_regressor: The main interface of NN models
  - init(): Define the NN architecture based on the modules in the encoders.py
  - train(): Call modules in the trainers to train itself
  - load_model(): 
  - predict(): Inference after training
- util: Support functions

**Customers**

One customer folder contains codes and data specific that to a customer. It typically 
contains:
- customer_data: Customer provided data
- customer_dataprocessing: Functions that created especially for processing customer data
- customer_train:  The main file for defining and 
training models. It will be the primary working file for a project.
- customer_inference: Load pretrained models for new predictions. The main delivery to a customer.
- customer_util: Support functions

-----------------------------------------------------------------------------------------------------------------------------
Try losses defined in losses_exp.py (proposed by Conditional Mutual information-based Contrastive Loss for Financial Time Series Forecasting (Hanwei Wu, Ather Gattami, Markus Flierl) for supervised constrative learning.
