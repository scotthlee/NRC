# Neural Record Captioning (NRC)
This repository contains code from the paper ["Natural Language Generation for Electronic Health Records"] (https://arxiv.org/abs/1806.01353).

## what's included
  1. Keras code for the NRC model.
  2. Training and testing scripts for the model.
  3. Example scripts for preprocessing EHR data to be used in the model.

## getting started
  1. Install the necessary Python modules (list below)
  2. Use `preprocessing/sparisfy.py` to convert the discrete variables in your EHRs to sparse format
  3. Use `preprocessing/words_to_integers.py` to conver your free text field to integers
  4. Train the autoencoder on the sparse records with `ae_training.py`
  5. Train the NRC model with `caption_training.py`
  6. Generate text with `caption_testing.py`

## required software
  1. Python 3.x
  1. Keras with the TensorFlow backend
  3. Pandas, NumPy, h5py, and scikit-learn

## hot tips
The default hyperparameters worked well for the data used in our paper, but they might not for yours, so feel free to experiment! Also,
we recommend a GPU for training the captioning model. We used a single NVIDIA Titan X for our experiments, and training
with ~2 million records took around 6 hours.

