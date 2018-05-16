import pandas as pd
import numpy as np
import itertools

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Embedding, LSTM, GRU
from keras.layers import Masking, TimeDistributed, Dropout, Reshape
from keras.regularizers import l1, l2
from scipy.sparse import save_npz, load_npz

import tools.text as tt
import tools.generic as tg
import tools.keras as tk

# Main class for the Neural Record Captioner
class NRC(object):
    def __init__(self,
                 sparse_size,
                 vocab_size,
                 max_length,
                 embedding_size,
                 hidden_size):
        self.sparse_size = sparse_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        return

    # Builds th training model for the captioner
    def build_training_model(self,
                             text_embedding=False,
                             trainable_records=True,
                             embeddings_dropout=0.2,
                             embeddings_reg=None,
                             recurrent_dropout=0.0,
                             encoding_layer=None):

        # Building a dense layer for the categorical variables in the record
        input_record = Input(shape=(self.sparse_size,), name='sparse_record')
        record_embedding_layer = Dense(units=self.embedding_size,
                                       name='record_embedding',
                                       trainable=trainable_records)
        if encoding_layer != None:
            ae_weights = encoding_layer.get_weights()
            record_embedding_layer = Dense(units=self.embedding_size,
                                           name='record_embedding',
                                           trainable=trainable_records,
                                           weights=ae_weights)
        embedded_record = record_embedding_layer(input_record)

        # Padding the record embedding and applying dropout
        reshaped_record = Reshape((1, self.embedding_size))(embedded_record)

        # Building an embedding layer for the free text in the record
        input_text = Input(shape=(self.max_length,), name='text_input')
        embedding_layer = Embedding(input_dim=self.vocab_size,
                                   output_dim=self.embedding_size,
                                   embeddings_regularizer=embeddings_reg,
                                   mask_zero=True,
                                   name='text_embedding')
        text_embedding = embedding_layer(input_text)

        # Building the RNN layer
        rnn = LSTM(units=self.hidden_size,
                   dropout=embeddings_dropout,
                   recurrent_dropout=recurrent_dropout,
                   return_sequences=True,
                   return_state=True,
                   name='rnn')

        # Zero state for the RNN layer
        batch_size = K.shape(input_record)[0]
        zero_state = [K.zeros((batch_size, self.hidden_size)),
                      K.zeros((batch_size, self.hidden_size))]

        # Running the record through the RNN first, and then the text
        rec_out, rec_h, rec_c = rnn(reshaped_record,
                                        initial_state=zero_state)
        rnn_output, _, _ = rnn(text_embedding,
                               initial_state=[rec_h, rec_c])

        # Adding a dense layer with softmax for getting predictions
        dense_layer = Dense(units=self.vocab_size,
                            activation='softmax',
                            name='dense_layer')
        output = dense_layer(rnn_output)
        inputs = [input_record, input_text]

        # Passing some layers up to the class
        self.rnn = rnn
        self.dense_layer = dense_layer
        self.text_embedding = embedding_layer
        self.record_lookup = Model(input_record, [rec_h, rec_c])
        self.training_model = Model(inputs=inputs, outputs=output)
        return

    # Loads a previously-trained training model; automatically builds
    # the inference model by default
    def load_training_model(self, mod_path):
        mod = load_model(mod_path)
        self.training_model = mod
        self.rnn = mod.layers[4]
        self.text_embedding = mod.layers[5]
        self.dense_layer = mod.layers[6]
        self.record_lookup = Model(mod.input[0],
                                   mod.layers[4].get_output_at(0)[1:])
        self.build_inference_model()
        return

    # Builds the inference model for the captioner using the embedding
    # layer, RNN, and dense layers from the training model
    def build_inference_model(self):

        # Defining inputs for the states of the encoder model
        input_h = Input(shape=(self.hidden_size,))
        input_c = Input(shape=(self.hidden_size,))
        input_states = [input_h, input_c]

        # Defining an input for the text sequence
        input_text = Input(shape=(self.max_length,),
                           name='input_text')
        text_embedding = self.text_embedding(input_text)

        # Running a step through the RNN
        rnn_output, output_h, output_c = self.rnn(text_embedding,
                                                  initial_state=input_states)
        output_states = [output_h, output_c]

        # Getting predictions from the dense layer
        softmax_probs = self.dense_layer(rnn_output)
        model = Model([input_text] + input_states,
                      [softmax_probs] + output_states)
        self.inference_model = model
        return

    # Gets the top K next predictions and their scores for a given sequence
    def beam_filter(self, seq, states, k=5, axis=1):
        previous = seq[np.where(seq != 0)]
        split_states = [states[0], states[1]]
        probs, h, c = self.inference_model.predict([seq] + split_states)
        best_next = tk.top_n(probs[0, len(previous)-1, :], k)
        best_seqs = np.array([np.concatenate([previous, next])
                              for next in best_next])
        beam_probs = np.array([tk.fetch_proba(probs, seq)
                               for seq in best_seqs])
        scores = tk.logsum(beam_probs, axis)
        states = list(itertools.repeat([h, c], k))
        out_seqs = np.array([tk.make_starter(self.max_length, seq)
                             for seq in best_seqs])
        return {'seqs':out_seqs, 'states':states, 'scores':scores}

    # Main function to generate captions from a given record
    def caption(self,
                record,
                vocab,
                level='word',
                method='greedy',
                k=5,
                temperature=1.0):
        # Fetching the bookend values
        if level == 'word':
            seed = vocab['start_string']
            end = vocab['end_string']
        elif level == 'char':
            end = vocab['.']
            seed = vocab['<']

        # Make a max_length array from the starting index in seed
        caption = tk.make_starter(self.max_length, [seed])
        seed_length = np.sum(seed != 0)

        # Getting the embedded version of the record to pass to the RNN
        feed_states = self.record_lookup.predict(record.toarray())

        # Generating captions with a beam search decoder
        if method == 'beam':
            # Running the first beam with the seed sequence
            seed_beam = self.beam_filter(caption, feed_states, k=k)

            # Making holders for the output and states, and setting an
            # endpoint for the beam search loop
            live_seqs = seed_beam['seqs']
            beam_states = seed_beam['states']
            dead_seqs = list([])
            search_end = self.max_length - 1

            # Running the main loop for the subsequent beams
            for i in range(seed_length, search_end):
                if live_seqs.shape[0]==0 or k==0:
                    break
                beams = list([])
                # Running a beam for each of the candidate sequences
                for i, seq in enumerate(live_seqs):
                    in_states = beam_states[i]
                    beams.append(self.beam_filter(seq, in_states, k=k))

                # Getting the states, sequences, and scores from each beam
                beam_states = np.concatenate([beam['states']
                                                for beam in beams], 0)
                beam_seqs = np.concatenate([beam['seqs']
                                            for beam in beams], 0)
                beam_scores = np.concatenate([beam['scores']
                                              for beam in beams], 0)

                # Saving the top k sequences from the list of candidates
                topk = tk.top_n(beam_scores, n=k, expand_dims=False)
                live_seqs = beam_seqs[topk, :]
                beam_states = beam_states[topk, :]

                # Finding finished sequences and trimming the live ones
                any_dead = np.any(live_seqs == end, axis=1)
                if np.any(any_dead):
                    where_dead = np.where(any_dead)[0]
                    current_dead = live_seqs[where_dead, :]
                    live_seqs = np.delete(live_seqs, where_dead, 0)
                    [dead_seqs.append(seq) for seq in current_dead]
                    k -= len(current_dead)

                # Adding any leftover live sequences to the dead ones
                if i == search_end:
                    [dead_seqs.append(seq) for seq in live_seqs]

            # Returning the finished sequences
            dead_seqs = np.array(dead_seqs)
            out = np.array([tt.ints_to_text(seq[0], vocab) for seq in dead_seqs])
            return out

        # Generating captions with (greedy) sampling
        if method != 'beam':
            for i in range(seed_length, self.max_length):
                probs, h, c = self.inference_model.predict([caption] +
                                                             feed_states)
                if method == 'greedy':
                    best = np.argmax(probs[0, i-1, :])
                elif method == 'sampling':
                    current_probs = probs[0, i-1, :]
                    melted_probs = np.log(current_probs) / temperature
                    exp_melted = np.exp(melted_probs)
                    new_probs = exp_melted / np.sum(exp_melted)
                    best = np.random.choice(range(self.vocab_size),
                                            p=new_probs)
                caption[0, i] = best
                feed_states = [h, c]
                if best == end:
                    break
            out = tt.ints_to_text(caption[0], vocab)
            return out
