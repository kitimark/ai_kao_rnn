import csv
import numpy as np
import deepcut

from keras.models import load_model

#------------------------ Read data -------------------------
TIMESTAMP = '1556386535.274951'
model = load_model('model/model'+TIMESTAMP+'.h5')
# print(model.input)

vocab = []
with open('result/list'+TIMESTAMP+'.csv', 'r', encoding='utf-8-sig') as f:
    vocab = list(csv.reader(f, delimiter=',', quotechar='|'))

sentences = []
with open('test/input.txt', 'r', encoding='utf-8-sig') as f:
    for line in f:
        sentences.append(line.rstrip().split('::')[1])

words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
# print(words)

# fix space
sen = []
for sentence in words:
    list = []
    for word in sentence:
        for w in word.split(' '):
            if w != '' and w != ',' and w != '!':
                try:
                    val = int(w)
                except ValueError:
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

# print(split_sen)

sentence_lengths = []
for sentence in split_sen:
    sentence_lengths.append(len(sentence))
    print(sentence)
max_length = max(sentence_lengths)
print(max_length)

words = split_sen

#---------------------------- Extract word vectors ----------------------------------

vocab = [w for s in vocab for w in s]

# write file list of words
with open('result/list_pred.txt', 'w', encoding='utf-8') as f:
    for line in vocab:
        # print(line)
        f.write(line + '\n')

with open('result/list_pred.csv', 'w', encoding='utf8', newline='') as f:
    f_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in vocab:
        f_writer.writerow([line])

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

#------------------------- Evaluation----------------------------------------------------
y_pred = model.predict(word_vectors)
print(y_pred)

#------------------------- Save -------------------------------------------------------------------------------
with open('result/ans.txt', 'w', encoding='utf-8-sig') as f:
    LABEL_NAMES = ['H', 'M', 'P']
    for i, line in enumerate(y_pred):
        f.writelines(str(i+1)+'::'+str(LABEL_NAMES[np.argmax(line)])+'\n')
    
