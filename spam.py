from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from numpy import zeros
from random import shuffle
from random import seed
from matplotlib import pyplot

'''
Read the file with the training and test data and return
it as two separate lists. Both lists will be shuffled before
they are returned.
'''
def read_lines():
    train_lines = []
    test_lines = []
    current_lines = []

    with open('SpamDetectionData.txt') as f: 
        for line in f.readlines(): 
            if line.startswith('# Test data', 0):
                train_lines = current_lines
                current_lines = test_lines
            elif line.startswith('#', 0):
                '''
                Ignore comment lines
                '''
            elif line == '\n':
                '''
                Ignore empty lines
                '''
            else:
                current_lines.append(line)

    test_lines = current_lines
    
    seed(1337)
    shuffle(train_lines)
    shuffle(test_lines)

    print('Read training lines: ', len(train_lines))
    print('Read test lines: ', len(test_lines))

    return train_lines, test_lines

'''
Take a list of lines from the original input file (train or test), remove
paragraphs and line breaks and split into label and data by using the comma 
as divider. Return as two separate lists preserving the sort order.
'''
def split_lines(lines):
    data = []
    labels = []
    maxtokens = 0
    for line in lines:
        label_part, data_part = line.replace('<p>','').replace('</p>','').replace('\n', '').split(',')
        data.append(data_part)
        labels.append(label_part)
        if (len(data_part)>maxtokens):
            maxtokens=len(data_part)

    print('maxlen ', maxtokens)

    return data, labels

'''
While processing the data with Keras each original text will converted
to a list of indices. These indices point to words in a dictionary
of all words contained in the training data. We convert this to a binary
matrix. The value 1 in the matrix says that a word (x in the matrix) is
contained in a given text (y in the matrix)
'''
def vectorize_sequences(sequences, dimension=4000):
    results = zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

'''
The label vectorization is quite simple:
  the value 1 is for spam,
  the value 0 is form ham
'''
def vectorize_labels(labels):
    results = zeros(len(labels))
    for i, label in enumerate(labels):
        if (label.lower() == 'spam'):
            results[i] = 1
    return results

def test_predict(model, testtext, expected_label):
    testtext_list = []
    testtext_list.append(testtext)
    testtext_sequence = tokenizer.texts_to_sequences(testtext_list)
    x_testtext = vectorize_sequences(testtext_sequence)
    prediction = model.predict(x_testtext)[0][0]
    
    print("Sentiment: %.3f" % prediction, 'Expected ', expected_label)

    if prediction > 0.5:
        if expected_label == 'Spam':
            return True
    else:
        if expected_label == 'Ham':
            return True
    
    return False

def plot_accuracy(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['training', 'validation'], loc='lower right')
    pyplot.show()

def plot_loss(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['training', 'validation'], loc='upper right')
    pyplot.show()

# Start script

# First split train lines from test lines
train_lines, test_lines = read_lines()

# Split data from label for each line
train_data_raw, train_labels_raw = split_lines(train_lines)
test_data_raw, test_labels_raw = split_lines(test_lines)

# Use Keras Tokenizer to vectorize text: 
# fit_on_texts will setup the internal vocabulary using all words
# from the training data and attaching indices to them
# texts_to_sequences will transform each text into sequence of
# integer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data_raw)
train_data_seq = tokenizer.texts_to_sequences(train_data_raw)
test_data_seq = tokenizer.texts_to_sequences(test_data_raw)

# Finally the integer sequenes are converted to a binary (numpy)
# matrix where rows are for the text lines, columns are for
# the words. 1 = word is inside text, 0 = word is not inside
x_train = vectorize_sequences(train_data_seq, 4000)
print('Lines of training data: ', len(x_train))
x_test = vectorize_sequences(test_data_seq, 4000)
print('Lines of test data: ', len(x_test))

# The labels are also converted to a binary vector.
# 1 means spam, 0 means ham
y_train = vectorize_labels(train_labels_raw)
print('Lines of training results: ', len(y_train))
y_test = vectorize_labels(test_labels_raw)
print('Lines of test results: ', len(y_test))

# Now we build the Keras model
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(4000,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# Train the model
history = model.fit(x_train,y_train,epochs=8,batch_size=100,validation_split=0.3)

# summarize history for accuracy
plot_accuracy(history)

# summarize history for loss
plot_loss(history)

# Evaluate the model
results = model.evaluate(x_test, y_test)
print(model.metrics_names)
print('Test result: ', results)

# Manual test over all test records
correct = 0
wrong = 0
for input_text, expected_label in zip(test_data_raw, test_labels_raw):
    if test_predict(model, input_text, expected_label):
        correct = correct + 1
    else:
        wrong = wrong + 1

print('Predictions correct ', correct, ', wrong ', wrong)


