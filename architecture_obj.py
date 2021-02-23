import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization, MaxPooling1D, ReLU, concatenate, PReLU
from tensorflow.keras.initializations import normal, orthogonal

def SNR(y_true, y_pred):
    """
    Compute the Signal to Noise Ratio which will be used as a metric for our Unet
    """

    cst_10 = tf.constant(10, dtype = tf.float32)

    fraction = tf.math.divide(tf.math.reduce_mean(tf.math.abs(y_true)),
                            tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_true,y_pred))))
    
    log_fraction = tf.math.divide(tf.math.log(fraction),tf.math.log(cst_10))

    return tf.math.multiply(log_fraction, cst_10)

def LSD(y_true, y_pred):
    """
    Compute the Log Spectral Distance of the true and reconstructed signal
    """

    cst_10 = tf.constant(10, dtype = tf.float32)

    Spec_true = tf.signal.stft(y_true, 2048, 512, window_fn = tf.signal.hamming_window)
    Spec_pred = tf.signal.stft(y_pred, 2048, 512, window_fn = tf.signal.hamming_window)

    Spec_true = tf.math.pow(tf.math.abs(Spec_true), tf.constant(2, dtype = tf.float32))
    Spec_pred = tf.math.pow(tf.math.abs(Spec_pred), tf.constant(2, dtype = tf.float32))

    Spec_true = tf.math.divide(tf.math.log(Spec_true), tf.math.log(cst_10))
    Spec_pred = tf.math.divide(tf.math.log(Spec_pred), tf.math.log(cst_10))

    inside_term = tf.math.reduce_mean(tf.math.pow(tf.math.subtract(Spec_true, Spec_pred), tf.constant(2, dtype = tf.float32)), axis = 1)
    
    return tf.math.reduce_mean(tf.math.sqrt(inside_term))

def AudioUnet(L, ndim):
  n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
  n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

  init_input = Input(shape = (ndim,1))

  downsampling_outputs = []

  #downsampling blocks
  x = Conv1D(filters = n_filters[0],
            kernel_size = n_filtersizes[0],
            padding = "same",
            kernel_initializer = "orthogonal")(init_input)
  x = MaxPooling1D()(x)
  x = LeakyReLU(0.2)(x)
  downsampling_outputs.append(x)

  for l, nf, nfs in zip(range(1,L), n_filters, n_filtersizes):
    x = Conv1D(filters = nf,
            kernel_size = nfs,
            padding = "same",
            kernel_initializer = "orthogonal")(x)
    x =  MaxPooling1D()(x)
    x = LeakyReLU(0.2)(x)
    downsampling_outputs.append(x)

  #bottleneck block
  x = Conv1D(filters = n_filters[-1],
            kernel_size = n_filtersizes[-1],
            padding = "same",
            kernel_initializer = "orthogonal")(x)
  x = MaxPooling1D()(x)
  x = Dropout(0.5)(x)
  x = LeakyReLU(0.2)(x)

  #upsampling blocks
  for l, nf, nfs, l_in in reversed(list(zip(list(range(L)), n_filters, n_filtersizes, downsampling_outputs))):
    x = Conv1D(filters = nf,
                kernel_size = nfs,
                padding = "same",
                kernel_initializer = "orthogonal")(x)
    x = Dropout(0.5)(x)
    x = ReLU()(x)
    x = SubPixel1D(x)
    x = concatenate([x, l_in], axis = 2)

  #final convolution layer
  x = Conv1D(filters = 2,
            kernel_size = 9,
            padding = "same",
            kernel_initializer = "normal")(x)

  x = SubPixel1D(x)

  x = Add()([x,init_input])

  return tf.keras.Model(inputs = init_input, outputs = x)

def SubPixel1D(I, r=2):
    """One-dimensional subpixel upsampling layer
    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r).
    Works with multiple channels: (B,L,rC) -> (B,rL,C)
    """
    with tf.name_scope('subpixel'):
        _, w, rc = I.get_shape()
        assert rc % r == 0
        c = rc / r
        X = tf.transpose(I, [2,1,0]) # (rc, w, b)
        X = tf.batch_to_space(X, [r], [[0,0]]) # (c, r*w, b)
        X = tf.transpose(X, [2,1,0])
        return X