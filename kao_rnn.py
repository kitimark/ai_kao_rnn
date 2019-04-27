import csv
import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense, GRU, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix

#------------------------- Read data ------------------------------
file = open('data/dataset.csv', 'r',encoding = 'utf-8-sig')
data = list(csv.reader(file, delimiter=',', quotechar='|'))
shuffle(data)

# for d in data:
#     print(d)

# labels = [int(d[0]) for d in data]
LABEL_NAMES = { "H" : 0, "M" : 1, "P" : 2}
labels = [LABEL_NAMES[d[0]] for d in data]
sentences = [d[1] for d in data]

words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]

# fix space
sen = []
for sentence in words:
    list = []
    for word in sentence:
        for w in word.split(' '):
            if w != '':
                list.append(w)
    sen.append(list)

# cut word
amount = 0
i_kao = []
for sentence in sen:
    flag = 0
    for i, word in enumerate(sentence):
        if word.find('เขา') != -1:
            flag += 1
            print(word, end=' ')
            print(i)
            i_kao.append(i)
    if flag != 1:
        print(sentence)

split_sen = []
for i, sentence in enumerate(sen):
    start = i_kao[i] - 10 if i_kao[i] - 10 > 0 else 0
    end = i_kao[i] + 10 if i_kao[i] + 10 < len(sentence) else len(sentence)
    
    split_sen.append(sentence[start:end])


sentence_lengths = []
for sentence in split_sen:
    sentence_lengths.append(len(sentence))
    print(sentence)
max_length = max(sentence_lengths)
print(max_length)

words = split_sen

#---------------------------- Extract word vectors ----------------------------------

vocab = set([w for s in sen for w in s])

# write file list of words
with open('result/list.txt', 'w', encoding='utf-8') as f:
    for line in vocab:
        # print(line)
        f.write(line + '\n')

pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
count = 0
vocab_vec = {}
for line in pretrained_word_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab):
            vocab_vec[line[0]] = line[1:]
    count = count + 1

word_vectors = np.zeros((len(words),max_length,300))
sample_count = 0
for s in words:
    word_count = 0
    for w in s:
        try:
            word_vectors[sample_count,19-word_count,:] = vocab_vec[w]
            word_count = word_count+1
        except:
            pass
    sample_count = sample_count+1

print(word_vectors.shape)
print(word_vectors[0])

#--------------- Create recurrent neural network-----------------
inputLayer = Input(shape=(20,300,))
rnn = GRU(30, activation='relu')(inputLayer)
rnn = Dropout(0.5)(rnn)
outputLayer = Dense(3, activation='softmax')(rnn)
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------- Train neural network-----------------------
history = model.fit(word_vectors, to_categorical(labels), epochs=300, batch_size=50, validation_split = 0.2)

#-------------------------- Evaluation-----------------------------
y_pred = model.predict(word_vectors[240:,:,:])

cm = confusion_matrix(labels[240:], y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()