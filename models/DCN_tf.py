import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization


class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation, isDropout=False):
        super().__init__()
        self.hidden_layer = []
        for x in hidden_units:
            self.hidden_layer.append(Dense(x, activation=activation))
            # self.hidden_layer.append(BatchNormalization())

            if isDropout:
                drop_n_units = 0.3
                print('drop out {} units'.format(drop_n_units))
                self.hidden_layer.append(Dropout(drop_n_units))

        self.output_layer = Dense(output_dim, activation=None)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output

class Cross_layer(Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super().__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)]

        self.cross_bias = [
            self.add_weight(name='b'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        xl = x0

        for i in range(self.layer_num):
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i])
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl

        output = tf.squeeze(xl, axis=2)
        return output


class DCN_tf(Model):
    def __init__(self,
                 feature_columns, hidden_units,
                 output_dim=16, activation='relu', layer_num=2,
                 isDropout=False, reg_w=1e-4, reg_b=1e-4):

        super().__init__()
        self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns, name='feat_layer')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense_layer = Dense_layer(hidden_units, output_dim, activation, isDropout)
        self.cross_layer = Cross_layer(layer_num, reg_w=reg_w, reg_b=reg_b)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs):
        x = self.feature_layer(inputs)
        # BN
        x = self.bn(x)
        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.dense_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
        return output