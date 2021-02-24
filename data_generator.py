class data_generator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim, n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    #preprocessing modules 
    def splice(self, song, d_f):
        """
        Dissecting the audio signals to chuncks of size dim (default 2048 as implied in paper)
        song: the full song audio signal for dissection
        d_f: downsampling factor
        """

        downsampled_data = []
        sample_length = self.dim

        #appending zeros to perfectly split the data at 2048 segments (without interfering with other speaker)
        zero_to_append = sample_length - song.size % sample_length
        song = np.append(song, np.zeros(zero_to_append))

        song_data.append(song)

        #downsampling (8 Khz Fs -> Max freq 4 Khz) and processing it with an anti alias filter
        song = song[::d_f]
        song = anti_alias(song)

        #cubic splines
        song = spline(song)

        downsampled_data.append(song)



    def anti_alias(numpydata):
        """
        Apply a 3rd degree Butterworth Low Pass filter for anti aliasing purposes.
        """
        b,a = signal.butter(3, Wn = 44100/8, fs = 44100)
        
        out = signal.filtfilt(b,a,numpydata)
        
        return out.astype(np.float32)

    def spline(data):
        """
        Apply the cubic spline interpolation for simulating a naive upsampling.
        """
        out = []
        
        data_in = data.numpy().astype(np.float32)
        
        for i in range(batch):
            
            x = np.arange(1,width + 1)
            cs = CubicSpline(x,data_in[i])
            
            y = np.arange(1,width+1, 0.25)
            
            out.append(cs(y))
            
        out = np.array(out)
        
        return out
    
    def preprocessing(self, X):
        pass

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

        return X