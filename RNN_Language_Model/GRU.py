# Multilayered (or singlelayered) GRU based RNN for text generation using Keras libraries

# import libraries
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import random
import sys
import csv
import os

# load Dataset
from pip._vendor.distlib.compat import raw_input

# path = raw_input(
#     "Enter file name (example: Wittgenstein.txt) for training and testing data (make sure it's in the same directory):\n ")
# dataset = open(path).read().lower()

dataset = ""
with open('Shakespeare_data.csv', 'rt') as f:
    reader = csv.reader(f, skipinitialspace=True)
    for x in reader:
        dataset += " " + x[5].lower()

print("Parsed %d sentences." % (len(dataset)))

# store the list of all unique characters in dataset
chars = sorted(list(set(dataset)))

total_chars = len(dataset)
vocabulary = len(chars)

print("Total Characters: ", total_chars)
print("Vocabulary: ", vocabulary)

# Creating dictionary or map in order to map all characters to an integer and vice versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

sample_len = 256
temperature = 1
Answer = 0

if Answer == 0:
    hidden_layers = 4
    neurons = []
    if hidden_layers == 0:
        hidden_layers = 1;
    for i in range(0, hidden_layers):
        neurons.append(40)
    seq_len = 40
    skip = 2
    skip = skip + 1
    learning_rate = 0.01
    dropout_rate = 0.2
    batch = 256

if Answer != 0:
    try:
        f = open('GRUModelInfoX', "r")
        lines = f.readlines()
        for i in lines:
            thisline = i.split(" ")
        seq_len = int(thisline[0])
        batch = int(thisline[1])
        skip = int(thisline[2])
        f.close()
    except:
        print("\nUh Oh! Caught some exceptions! May be you are missing the file having time step information")

if Answer == 0 or Answer == 2:
    dataX = []
    dataY = []

    for i in range(0, total_chars - seq_len, skip):  # Example of an extract of dataset: Language
        dataX.append(dataset[i:i + seq_len])  # Example Input Data: Languag
        dataY.append(dataset[i + seq_len])  # Example of corresponding Target Output Data: e

    total_patterns = len(dataX)
    print("\nTotal Patterns: ", total_patterns)

    # One Hot Encoding...
    X = np.zeros((total_patterns, seq_len, vocabulary), dtype=np.bool)
    Y = np.zeros((total_patterns, vocabulary), dtype=np.bool)

    for pattern in range(total_patterns):
        for seq_pos in range(seq_len):
            vocab_index = char_to_int[dataX[pattern][seq_pos]]
            X[pattern, seq_pos, vocab_index] = 1
        vocab_index = char_to_int[dataY[pattern]]
        Y[pattern, vocab_index] = 1

if Answer == 0:
    print('\nBuilding model...')

    model = Sequential()
    if hidden_layers == 1:
        model.add(GRU(neurons[0], input_shape=(seq_len, vocabulary)))
    else:
        model.add(GRU(neurons[0], input_shape=(seq_len, vocabulary), return_sequences=True))
    model.add(Dropout(dropout_rate))
    for i in range(1, hidden_layers):
        if i == (hidden_layers - 1):
            model.add(GRU(neurons[i]))
        else:
            model.add(GRU(neurons[i], return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(vocabulary))
    model.add(Activation('softmax'))

    RMSprop_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop_optimizer)

    # save model information
    model.save('GRUModelX.h5')
    f = open('GRUModelInfoX', 'w+')
    f.write(str(seq_len) + " " + str(batch) + " " + str(skip))
    f.close()

else:
    print('\nLoading model...')
    try:
        model = load_model('GRUModelX.h5')
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved model to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)

model.summary()

# define the checkpoint
filepath = "BestGRUWeightsX.h5"  # Best weights for sampling will be saved here.
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,
                             mode='min')


# Function for creating a sample text from a random seed (an extract from the dataset).
# The seed acts as the input for the GRU RNN and after feed forwarding through the network it produces the output
# (the output can be considered to be the prediction for the next character)
# gntext = ''
def sample(seed):
    for i in range(sample_len):
        # One hot encoding the input seed
        x = np.zeros((1, seq_len, vocabulary))
        for seq_pos in range(seq_len):
            vocab_index = char_to_int[seed[seq_pos]]
            x[0, seq_pos, vocab_index] = 1
        prediction = model.predict(x, verbose=0)


        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature

        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)

        RNG_int = np.random.choice(range(vocabulary), p=prediction.ravel())

        next_char = int_to_char[RNG_int]

        # Display the chosen character
        sys.stdout.write(next_char)
        # gntext += next_char
        # print(next_char)
        sys.stdout.flush()
        # modifying seed for the next iteration for finding the next character
        seed = seed[1:] + next_char

    print()


if Answer == 0 or Answer == 2:
    if Answer == 2:
        filename = "GRUWeightsX.h5"
        try:
            model.load_weights(filename)
        except:
            print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved weights to load.")
            print("Solution: May be create and train the model anew ?")
            sys.exit(0)
    # Train Model and print sample text at each epoch.
    for iteration in range(1, 2):
        print()
        print('Iteration: ', iteration)
        print()

        # Train model. If you have forgotten: X = input, Y = targeted outputs
        model.fit(X, Y, batch_size=batch, nb_epoch=1, callbacks=[checkpoint])
        model.save_weights(
            'GRUWeightsX.h5')  # Saving current model state so that even after terminating the program; training
        # can be resumed for last state in the next run.
        print()

        # Randomly choosing a sequence from dataset to serve as a seed for sampling
        start_index = random.randint(0, total_chars - seq_len - 1)
        seed = dataset[start_index: start_index + seq_len]

        sample(seed)
else:
    # load the network weights
    filename = "BestGRUWeightsX.h5"
    try:
        model.load_weights(filename)
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved weights to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)
    Answer2 = "y"
    while Answer2 == "y" or Answer2 == "Y":
        print("\nGenerating Text:\n")
        # Randomly choosing a sequence from dataset to serve as a seed for sampling
        start_index = random.randint(0, total_chars - seq_len - 1)
        seed = dataset[start_index: start_index + seq_len]
        sample(seed)
        print()
        Answer2 = raw_input("Generate another sample Text? (y/n): ")