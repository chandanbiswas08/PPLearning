from mnist_data_helper import *

class LR(tf.keras.Model):
    def __init__(self, NUM_CLASS):
        super(LR, self).__init__()
        self.dense_last = tf.keras.layers.Dense(NUM_CLASS, activation='sigmoid', name='dense_last')

    def call(self, inputs):
        output = self.dense_last(inputs)
        return output


class RowEncoder(tf.keras.Model):
    def __init__(self):
        super(RowEncoder, self).__init__(name='SE_CNN')
        self.dense = tf.keras.layers.Dense(Config.SE_DENSE_DIM, name='SE_dense', activation='relu')
        self.conv1D = tf.keras.layers.Conv1D(Config.SE_CNN_FILTER, 1, padding='valid', activation='relu', strides=1)
        self.global_max_pooling1D = tf.keras.layers.GlobalMaxPooling1D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        net = tf.transpose(inputs, [0, 2, 1])
        net = self.dense(net)
        net = self.conv1D(net)
        en_out = self.global_max_pooling1D(net)
        return en_out


class Mean(tf.keras.layers.Layer):
    def __init__(self):
        super(Mean, self).__init__()

    def call(self, inputs, max_imp_words):
        return tf.reduce_sum(inputs, axis = 1) / float(max_imp_words)

class Concatenate(tf.keras.layers.Layer):
    """
    Layer for concatenation.
    """
    def __init__(self):
        super(Concatenate, self).__init__()

    def call(self, inputs):
        input1, input2 = inputs
        input1 = tf.expand_dims(input1, axis=-2)
        # [batchsize, 1, input1_dim]
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis=-1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)


class construct_gumbel_selector(tf.keras.Model):
    def __init__(self):
        super(construct_gumbel_selector, self).__init__()
        self.row_encoder = RowEncoder()
        self.image_encoder = tf.keras.layers.TimeDistributed(self.row_encoder)
        self.conv_layer_1 = tf.keras.layers.Conv1D(Config.CG_CNN_FILTER[0], 3, padding='same', activation='relu', strides=1, name='conv1_gumbel')
        self.global_max_pooling1D = tf.keras.layers.GlobalMaxPooling1D(name='new_global_max_pooling1d_1')
        self.dense_1 = tf.keras.layers.Dense(Config.CG_DENSE_DIM, name='new_dense_1', activation='relu')
        self.conv_layer_2 = tf.keras.layers.Conv1D(Config.CG_CNN_FILTER[1], 3, padding='same', activation='relu', strides=1, name='conv2_gumbel')
        self.conv_layer_3 = tf.keras.layers.Conv1D(Config.CG_CNN_FILTER[2], 3, padding='same', activation='relu', strides=1, name='conv3_gumbel')
        self.conv_layer_4 = tf.keras.layers.Conv1D(Config.CG_CNN_FILTER[3], 1, padding='same', activation='relu', strides=1, name='conv_last_gumbel')
        self.conv_layer_last = tf.keras.layers.Conv1D(Config.CG_CNN_FILTER[4], 1, padding='same', activation=None, strides=1, name='conv4_gumbel')
        self.concat = Concatenate()

    def call(self, inputs):
        net = self.image_encoder(inputs)
        first_layer = self.conv_layer_1(net)

        # global info
        net_new = self.global_max_pooling1D(first_layer)
        global_info = self.dense_1(net_new)

        # local info
        net = self.conv_layer_2(first_layer)
        local_info = self.conv_layer_3(net)
        combined = Concatenate()([global_info, local_info])
        net = tf.keras.layers.Dropout(0.2, name='new_dropout_2')(combined)
        net = self.conv_layer_4(net)
        logits_T = self.conv_layer_last(net)
        return logits_T

class Sample_Concrete(tf.keras.layers.Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.
    """
    def __init__(self, **kwargs):
        self.tau0 = 0.5
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits, imp_row_count, row_count, TRAIN):

        if TRAIN == True:
            logits_ = tf.keras.layers.Permute((2, 1))(logits)
            unif_shape = tf.shape(logits_)[0]
            uniform = tf.random.uniform(shape =(unif_shape, imp_row_count, row_count), minval = np.finfo(tf.float32.as_numpy_dtype).tiny, maxval = 1.0)
            gumbel = - tf.cast(tf.math.log(-tf.math.log(uniform)), tf.float64)
            noisy_logits = (gumbel + logits_)/self.tau0
            samples = tf.nn.softmax(noisy_logits)
            samples = tf.math.reduce_max(samples, axis = 1)
            samples = tf.expand_dims(samples, -1)
            return samples
        else:
            logits_ = tf.squeeze(logits,axis=-1)
            threshold = tf.expand_dims(tf.nn.top_k(logits_, imp_row_count, sorted=True)[0][:, -1], -1)
            discrete_logits = tf.cast(tf.greater_equal(logits_, threshold), tf.float64)
            discrete_logits = tf.where(tf.equal(discrete_logits, True), logits_, 0)
            discrete_logits = tf.expand_dims(discrete_logits, axis=-1)
            return discrete_logits

class L2X(tf.keras.Model):
    def __init__(self):
        super(L2X, self).__init__()
        self.construct_gumbel_selector = construct_gumbel_selector()
        self.row_encoder = RowEncoder()
        self.image_encoder = tf.keras.layers.TimeDistributed(self.row_encoder, name='L2X_review_encoder')
        self.Sample_Concrete = Sample_Concrete()
        self.dense_1 = tf.keras.layers.Dense(Config.ENCODING_DIM, name='L2X_Fc', activation='relu')  # originally 250
        self.Mean = Mean()

    def call(self, inputs, imp_row_count, row_count, TRAIN):
        logits_T = self.construct_gumbel_selector(inputs)
        priority_weight = tf.squeeze(tf.keras.layers.Permute((2, 1))(logits_T), axis=1)
        T = self.Sample_Concrete(logits_T, imp_row_count, row_count, TRAIN)  # N,MRL,1
        image_en_out = self.image_encoder(inputs)  # N,MRL,250
        selected_encoding = tf.keras.layers.Multiply()([image_en_out, T])  # N,MRL,1
        net = self.Mean(selected_encoding, imp_row_count)
        l2x_out = self.dense_1(net)
        return l2x_out, T, priority_weight
