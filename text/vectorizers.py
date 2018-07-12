# -*- coding: utf-8 -*-
from collections import Counter
from tqdm import tqdm

class Vectorizer:
    prefix = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.idx2word = list()
        self.word2idx = dict()
        self.pretrained_embed_dir = config.pretrained_embed_dir

    def build_vectorizer(self, lines):
        self.word2idx, self.idx2word = self._build_vocab(lines)

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx['<UNK>']

    def _build_vocab(self, lines):
        counter = list()
        print("Build Vocabulary...")
        for line in tqdm(lines):
            counter += self.tokenizer.tokenize(line)
        counter = Counter(counter)
        print("Vocabulary Built. Total number of unique tokens : ", len(counter))
        idx2word = self.prefix + sorted([word for (word, freq) in counter.most_common(self.vocab_size-4)])
        word2idx = {word:idx for idx, word in enumerate(idx2word)}
        return word2idx, idx2word

    def state_dict(self):
        state = {'idx2word' : self.idx2word,
                 'word2idx' : self.word2idx}
        return state

    def load_state_dict(self, state_dict):
        self.idx2word = state_dict['idx2word']
        self.word2idx = state_dict['word2idx']
