import numpy as np
from text import normalizers, tokenizers, vectorizers


class Preprocessor:
    def __init__(self, config):
        self.min_length = config.min_length
        self.max_length = config.max_length
        Normalizer = getattr(normalizers, config.normalizer)
        Tokenizer = getattr(tokenizers, config.tokenizer)
        self.normalizer = Normalizer(config)
        self.tokenizer = Tokenizer(config)
        self.vectorizer = vectorizers.Vectorizer(self.tokenizer, config)

    def build_preprocessor(self, lines):
        self.vectorizer.build_vectorizer(lines)
        self.feature_extractors = list()

    def _preprocess(self, sentence):
        normalized_sentence = self.normalizer.normalize(sentence)
        tokenized_sentence = ["<SOS>"] + self.tokenizer.tokenize(normalized_sentence) + ["<EOS>"]
        indexed_sentence = [self.vectorizer.indexer(token) for token in tokenized_sentence]
        return indexed_sentence

    def preprocess(self, sentence):
        indexed_sentence = self._preprocess(sentence)
        padded_sentence = pad_sequences([indexed_sentence], maxlen=self.max_length)[0]
        return padded_sentence

class DynamicPreprocessor:
    def __init__(self, config):
        self.min_length = config.min_length
        self.max_length = config.max_length
        Normalizer = getattr(normalizers, config.normalizer)
        Tokenizer = getattr(tokenizers, config.tokenizer)
        self.normalizer = Normalizer(config)
        self.tokenizer = Tokenizer(config)
        self.vectorizer = vectorizers.Vectorizer(self.tokenizer, config)

    def build_preprocessor(self):
        self.vectorizer.build_vectorizer()

    def _preprocess(self, sentence):
        normalized_sentence = self.normalizer.normalize(sentence)
        tokenized_sentence = ["<SOS>"] + self.tokenizer.tokenize(normalized_sentence) + ["<EOS>"]
        indexed_sentence = [self.vectorizer.indexer(token) for token in tokenized_sentence] 
        return indexed_sentence

    def preprocess(self, sentence):
        indexed_sentence = self._preprocess(sentence)
        length = len(indexed_sentence)
        padded_sentence = pad_sequences([indexed_sentence], maxlen=self.max_length)[0]
        return padded_sentence, length

# from tflearn.data_utils import pad_sequences
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.
    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)
    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
