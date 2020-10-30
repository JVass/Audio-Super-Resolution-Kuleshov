32
import tensorflow as tf

def model(block_num, input_size):
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

    
    #---------- Upsampling Layer #1 -----------------
    U1_conv = tf.keras.layers.Conv1D(filters = 2048,
                                    kernel_size = 8,
                                    strides = 2,
                                    activation = "relu",
                                    padding = "same")(stacked_1)

    U1_dropout = tf.keras.layers.Dropout(rate = 0.4)(U1_conv)
    U1_ReLU = tf.keras.layers.ReLU()(U1_dropout)
    U1_
    #---------- Upsampling Layer #1 -----------------

    #---------- Upsampling Layer #1 -----------------

    #---------- Upsampling Layer #1 -----------------

