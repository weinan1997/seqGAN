import torch
import re


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        words = sentence.split(' ')
        for word in words:
            self.addWord(word)
        return words

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def normalizeString(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def readFile(self, path):
        print("Reading lines...")
        with open(path) as f:
            lines = f.read().strip().split('\n')
        lines = [self.normalizeString(l) for l in lines]
        print("Finish reading!")
        return lines

    def tokenize(self, lines, maxLength):
        print("Tokenizing...")
        tokenized_texts = []
        for line in lines:
            tokenized_texts.append(self.addSentence(line))
        print("Finish tokenizeing!")
        indexed_texts = []
        for line in tokenized_texts:
            indexed_words = [0]
            for word in line:
                indexed_words.append(self.word2index[word])
            indexed_words.append(1)
            if len(indexed_words) > maxLength:
                indexed_words = indexed_words[:maxLength]
                indexed_words[-1] = 1
            elif len(indexed_words) < maxLength:
                indexed_words += [2]*(maxLength-len(indexed_words))
            indexed_texts.append(indexed_words)
        return indexed_texts


lang = Lang()
lines = lang.readFile('data/image_coco.txt')
print('Number of sentences: ', len(lines))
indexed_texts = lang.tokenize(lines, 40)
print('Vocabulary size: ', len(lang.word2index.values()))
indexed_texts = torch.tensor(indexed_texts)
print(indexed_texts[0:2])
torch.save(indexed_texts, 'data/image_coco.data')
