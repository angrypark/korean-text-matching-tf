import re
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """
    Base class for text normalizer
    """
    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, sentence):
        normalized_text = sentence
        return normalized_text

class DummyNormalizer(BaseNormalizer):
    """
    A dummy normalizer which does nothing.
    """
    def __init__(self, config):
        pass

    def normalize(self, sentence):
        normalized_sentence = sentence
        return normalized_sentence


class BasicNormalizer(BaseNormalizer):
    def __init__(self, config):
        pass

    def normalize(self, sentence):
        normalized_sentence = delete_quote(sentence)
        normalized_sentence = sibalizer(normalized_sentence)
        return normalized_sentence


class AdvancedNormalizer(BaseNormalizer):
    def __init(self, config):
        pass

    def normalize(self, sentence):
        normalized_sentence = sentence.lower()
        normalized_sentence = delete_quote(normalized_sentence)
        normalized_sentence = sibalizer(normalized_sentence)
        normalized_sentence = bad_words_exchanger(normalized_sentence)
        return normalized_sentence


def delete_quote(sentence):
    sentence = sentence.replace("'", '').replace('"', '').strip()
    if sentence.find("10자") > -1:
        sentence = sentence[:sentence.find("10자")]
    return sentence


def sibalizer(sentence):
    r = re.compile('씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빨|\
씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}벌|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}뻘|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}펄|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|신[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}방|\
ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}ㅂ|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔')
    for sibal in r.findall(sentence):
        sentence = sentence.replace(sibal, "시발")
    return sentence


def bad_words_exchanger(sentence):
    bad_words = {'쓰레기': {'쓰레기', '쓰래기', 'ㅆㄹㄱ', '쓰렉', '쓰뤠기', '레기', '쓰렉이'},
                 '병신': {'병신', '븅신', '빙신', 'ㅂㅅ'},
                 '존나': {'존나', '졸라', '조낸', '존내', 'ㅈㄴ', '존니', '좆나', '좆도', '좃도', '좃나'},
                 '뻐큐': {'뻐큐', '뻑큐', '凸'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(sentence)
        for token in tokens:
            sentence = sentence.replace(token, bad_word)
    return sentence
