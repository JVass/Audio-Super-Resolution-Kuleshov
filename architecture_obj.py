import tensorflow as tf

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

    inside_term = tf.math.reduce_mean(tf.math.pow(tf.math.subtract(Spec_true, Spec_pred), tf.constant(2, dtype = tf.float32)))
    
    return tf.math.reduce_mean(tf.math.sqrt(inside_term))


#--------------- Architecture 4 Downsampling 4 Upsampling ---------------#
inputs = tf.keras.Input(shape(input_size,))

D1_conv = tf.keras.layers.Conv1D(filters = 512,
                                kernel_size = 64,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(inputs)

D2_conv = tf.keras.layers.Conv1D(filters = 512,
                                kernel_size = 32,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(D1_conv)

D3_conv = tf.keras.layers.Conv1D(filters = 512,
                                kernel_size = 16,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(D2_conv)

D4_conv = tf.keras.layers.Conv1D(filters = 1024,
                                kernel_size = 9,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(D3_conv)

#In the figure there is a bottleneck layer but in the paper Kuleshov doesnt provide us with info

U1_conv = tf.keras.layers.Conv1D(filters = 2048,
                                kernel_size = 8,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(stacked_1)

U1_dropout = tf.keras.layers.Dropout(rate = 0.4)(U1_conv)
U1_ReLU = tf.keras.layers.ReLU()(U1_dropout)
U1_DimShuffle = SubPixel1D(U1_ReLU)
U1_Stacking = tf.concat([U1_DimShuffle,D4_conv], 2)

U2_conv = tf.keras.layers.Conv1D(filters = 2048,
                                kernel_size = 8,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(U1_Stacking)

U2_dropout = tf.keras.layers.Dropout(rate = 0.4)(U2_conv)
U2_ReLU = tf.keras.layers.ReLU()(U2_dropout)
U2_DimShuffle = SubPixel1D(U2_ReLU)
U2_Stacking = tf.concat([U2_DimShuffle,D4_conv], 2)

U3_conv = tf.keras.layers.Conv1D(filters = 2048,
                                kernel_size = 8,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(U2_Stacking)

U3_dropout = tf.keras.layers.Dropout(rate = 0.4)(U3_conv)
U3_ReLU = tf.keras.layers.ReLU()(U3_dropout)
U3_DimShuffle = SubPixel1D(U3_ReLU)
U3_Stacking = tf.concat([U3_DimShuffle,D4_conv], 2)

U4_conv = tf.keras.layers.Conv1D(filters = 2048,
                                kernel_size = 8,
                                strides = 2,
                                activation = "relu",
                                padding = "same")(U3_Stacking)

U4_dropout = tf.keras.layers.Dropout(rate = 0.4)(U4_conv)
U4_ReLU = tf.keras.layers.ReLU()(U4_dropout)
U4_DimShuffle = SubPixel1D(U4_ReLU)
U4_Stacking = tf.concat([out,D4_conv], 2)

#Logically the paper implies that a final layer is
out = tf.keras.layers.Dense(1, activation = "tanh")(U4_Stacking)

#instantiate the model
model = tf.keras.Model(inputs = inputs, output = outputs)

