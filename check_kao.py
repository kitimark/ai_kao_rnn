import csv
import deepcut

sentences = []

with open('test/input.txt', 'r', encoding='utf8') as f:
    for line in f:
        y = line.split('::')
        sentences.append(y[1].rstrip())

# labels = [int(d[0]) for d in data]

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