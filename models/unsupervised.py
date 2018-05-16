from keras.models import Model
from keras.layers import Input, Dense

# A vanilla autoencoder
def autoencoder(sparse_dim, embedding_dim):
    input_record = Input(shape=(sparse_dim,))
    encoded = Dense(embedding_dim,
                    activation='relu',
                    name='encoder')(input_record)
    decoded = Dense(sparse_dim,
                    activation='sigmoid',
                    name='decoder')(encoded)
    model = Model(input_record, decoded)
    return model
