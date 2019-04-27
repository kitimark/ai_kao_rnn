import csv

labels = []
sentences = []

with open('data/ans.txt', mode='r', encoding='utf8') as f:
  for line in f:
    y = line.split('::')
    
    labels.append(y[1].rstrip())

    # if y[0] != str(i + 1):
    #   print(i + 1)
    # y = line.split()
    # print(i + 1)

print(labels)


# data = open('dataset/input.txt', mode='r', encoding="utf8")
# print(data.read())

with open('data/input.txt', 'r', encoding='utf8') as f:
  for line in f:
    y = line.split('::')
    sentences.append(y[1].rstrip())

print(sentences)

with open('data/dataset.csv', 'w', encoding='utf8', newline='') as f:
  f_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for i in range(len(labels)):
    label = labels[i]
    sentence = sentences[i]
    f_writer.writerow([label, sentence])