from jamo import h2j, j2hcj
import re
from konlpy.tag import Twitter
import sentencepiece as spm
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.word import WordExtractor
from collections import Counter
from abc import *


class BaseTokenizer(ABC):
    """
    Base class for tokenizers
    """
    def __init__(self, config):
        pass

    def fit(self, raw_text):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass

    @abstractmethod
    def tokenize(self, sentence):
        tokenized_sentence = sentence.split(" ")
        return tokenized_sentence
    

class DummyTokenizer(BaseTokenizer):
    def tokenize(self, sentence):
        tokenized_sentence = sentence.split(" ")
        return tokenized_sentence


class JamoTokenizer(BaseTokenizer):
    """
    Split text into jamos, delete all whitespace
    """
    def __init__(self, config):
        pass

    def tokenize(self, sentence):
        tokenized_sentence = [j for j in j2hcj(h2j(sentence))]
        return tokenized_sentence


class SyllableTokenizer(BaseTokenizer):
    """
    Split text into syllables.
    """
    def __init__(self, config):
        pass
    
    def tokenize(self, sentence):
        return sentence
    
    
class TwitterTokenizer(BaseTokenizer):
    """
    Tokenize text using Twitter of KoNLPy
    """
    def __init__(self, config):
        self.twitter = Twitter()

    def tokenize(self, sentence, stem=False, norm=False):
        tokenized_sentence = self.twitter.pos(sentence, stem=stem, norm=norm)
        tokenized_sentence = [token for token, pos in sentence]
        return tokenized_sentence


class SoyNLPTokenizer(BaseTokenizer):
    """
    Tokenize text using MaxScoreTokenizer of SoyNLP
    """
    def __init__(self, config):
        with open(config.soynlp_scores, "r") as f:
            scores = [line.strip().split("\t") for line in f]
            scores = {word:float(score) for word, score in scores}
        self.tokenizer = MaxScoreTokenizer(scores=scores)
    
    def tokenize(self, sentence):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        return tokenized_sentence
    

class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(config.sent_piece_model)
    
    def tokenize(self, sentence):
        tokens = self.tokenizer.EncodeAsPieces(sentence)
        tokens = [token.decode("utf-8").replace("▁","") for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens