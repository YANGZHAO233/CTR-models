import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding
import pandas as pd


class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        output = self.out_layer(inputs)
        return output

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim=1, activation='relu', dropout=0.0):
        super(Dense_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(output_dim, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.dropout(x)
        output = self.out_layer(x)
        return output

class CIN(Layer):
    def __init__(self, cin_size):
        super(CIN, self).__init__()
        self.cin_size = cin_size

    def build(self, input_shape):
        self.field_num = [input_shape[1]] + self.cin_size

        self.cin_W = [self.add_weight(
                         name='w'+str(i),
                         shape=(1, self.field_num[0]*self.field_num[i], self.field_num[i+1]),
                         initializer=tf.initializers.glorot_uniform(),
                         regularizer=tf.keras.regularizers.l1_l2(1e-5),
                         trainable=True)
                      for i in range(len(self.field_num)-1)]

    def call(self, inputs, **kwargs):
        k = inputs.shape[-1]
        res_list = [inputs]
        X0 = tf.split(inputs, k, axis=-1)
        for i, size in enumerate(self.field_num[1:]):
            Xi = tf.split(res_list[-1], k, axis=-1)
            x = tf.matmul(X0, Xi, transpose_b=True)
            x = tf.reshape(x, shape=[k, -1, self.field_num[0]*self.field_num[i]])
            x = tf.transpose(x, [1, 0, 2])
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding='VALID')
            x = tf.transpose(x, [0, 2, 1])
            res_list.append(x)

        res_list = res_list[1:]
        res = tf.concat(res_list, axis=1)
        output = tf.reduce_sum(res, axis=-1)
        return output


class xDeepFM(Model):
    def __init__(self, feature_columns, cin_size, hidden_units, output_dim=1, activation='relu', dropout=0.0):
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns]
        self.bn = tf.keras.layers.BatchNormalization()
        self.linear = Linear()
        self.dense_layer = Dense_layer(hidden_units, output_dim, activation, dropout)
        self.cin_layer = CIN(cin_size)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.bn(inputs)
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        # linear
        linear_out = self.linear(x)

        emb = [self.embed_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        emb = tf.transpose(tf.convert_to_tensor(emb), [1, 0, 2])

        # CIN
        cin_out = self.cin_layer(emb)

        # dense
        emb = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2]))
        emb = tf.concat([dense_inputs, emb], axis=1)
        dense_out = self.dense_layer(emb)

        output = self.out_layer(linear_out + cin_out + dense_out)
        return tf.nn.sigmoid(output)