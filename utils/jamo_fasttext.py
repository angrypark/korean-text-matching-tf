import os
import pickle
import sys
from datetime import datetime

from fasttext import fasttext
from gensim.models.wrappers.fasttext import FastText

from modules.nlp_pipeline.processor import JamoToWordProcessor, WordToJamoProcessor
from modules.nlp_pipeline.nlp_pipeline import NLPPipeline


class JamoFastText:
    """한국어로 된 문서의 자음과 모음을 분리하여 FastText를 적용합니다"""

    def __init__(self, postfix='_jamo'):
        """Initialize

        :param postfix: 자음과 모음이 분리된 corpus의 postfix
        """
        self.fasttext = None
        self.jamo_to_word_pipeline = NLPPipeline([
            JamoToWordProcessor()
        ])
        self.word_to_jamo_pipeline = NLPPipeline([
            WordToJamoProcessor()
        ])
        self.postfix = postfix

    def _generate_jamo_corpus(self, fname, input_pair):
        """자음과 모움이 분리된 corpus를 만듭니다

        :param fname: str, corpus file name
        :param input_pair: bool, input의 pair 여부
        """
        with open(fname, 'r') as file_r:
            with open(fname + self.postfix, 'w') as file_w:
                for line in file_r:
                    line = line.strip()
                    if input_pair:
                        line = line.split('\t')
                    else:
                        line = [line]

                    for sent in line:
                        file_w.write('%s\n' % self._convert_word_to_jamo(sent))

    @staticmethod
    def _del_jamo_corpus(fname):
        """자음과 모임이 분리된 corpus 삭제"""
        os.remove(fname)

    def train(self, input_txt, output_path, model='skipgram', lr=0.1, lr_update_rate=100, dim=100, ws=5, epoch=10,
              min_count=10, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6, thread=8, t=1e-4, input_pair=False,
              remove=True):
        """Train FastText model

        :param input_txt: corpus file name
        :param output_path: 결과 파일 경로
        :param model: 모델 종류, skipgram or cbow
        :param lr: learning rate
        :param lr_update_rate: learning rate update rate
        :param dim: dimension of embedding
        :param ws: window size
        :param epoch: epoch
        :param min_count: word min count
        :param neg:
        :param loss:
        :param bucket:
        :param minn: min ngram
        :param maxn: max ngram
        :param thread: number of threads
        :param t:
        :param input_pair: bool, input의 pair 여부
        :param remove: bool, 중간 파일 삭제 여부
        """

        self._generate_jamo_corpus(input_txt, input_pair=input_pair)

        input_txt += self.postfix
        if model == 'skipgram':
            ft = fasttext.skipgram(input_txt, output_path, lr=lr, dim=dim, ws=ws, epoch=epoch, min_count=min_count,
                                   neg=neg, loss=loss, bucket=bucket, minn=minn, maxn=maxn, thread=thread, t=t,
                                   lr_update_rate=lr_update_rate)
        elif model == 'cbow':
            ft = fasttext.cbow(input_txt, output_path, lr=lr, dim=dim, ws=ws, epoch=epoch, min_count=min_count,
                               neg=neg, loss=loss, bucket=bucket, minn=minn, maxn=maxn, thread=thread, t=t,
                               lr_update_rate=lr_update_rate)
        else:
            raise ValueError('model type must be either skipgram or cbow.')

        if remove:
            self._del_jamo_corpus(input_txt)

        self.load(output_path)

    def save_dict(self, fname, topn=100, verbose=True):
        """학습된 FastText를 dict 형태로 저장

        :param fname: 저장할 경로
        :param topn: 상위 n개만을 저장
        :param verbose:
        """
        if self.fasttext is None:
            raise Exception('Train fasttext first')

        word_to_similar = {}

        words = [self.jamo_to_word_pipeline.run(word) for word in self.fasttext.fasttext.wv.vocab]

        s_time = datetime.now()
        for n, word in enumerate(words):
            if verbose:
                if n % 10 == 0:
                    remain_time = ((datetime.now() - s_time) / (n + 1)) * (len(words) - n)
                    sys.stdout.write('\r [%d / %d] remain_time: %f min' %
                                     (n, len(words), remain_time.total_seconds() / 60))

            word_to_similar[word] = []
            for w, dist in self.most_similar(word, topn=topn):
                word_to_similar[word].append((w, dist))
        sys.stdout.write('\r Time to save dict: %f min' % ((datetime.now()-s_time).total_seconds() / 60))

        with open(fname, 'wb') as f:
            pickle.dump(word_to_similar, f)

    def load(self, fname):
        self.fasttext = FastText.load_fasttext_format(fname)

    def most_similar(self, positive=None, negative=None, topn=10):
        """gensim의 most_similar와 같은 역할

        :param positive:
        :param negative:
        :param topn:
        :return:
        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.fasttext.wv.init_sims()
        similars = self.fasttext.most_similar(positive=self._convert_word_to_jamo(positive),
                                              negative=self._convert_word_to_jamo(negative),
                                              topn=topn)
        return [(self.jamo_to_word_pipeline.run(jamo), dist) for jamo, dist in similars]

    def _convert_word_to_jamo(self, words):
        """들어온 단어 혹은 단어 리스트의 자모를 분리

        :param words: 단어 혹은 단어 리스트
        :return: 자음과 모음이 분리된 단어 혹은 단어 리스트
        """
        if isinstance(words, str):
            return self.word_to_jamo_pipeline.run(words)
        elif isinstance(words, list):
            return [self.word_to_jamo_pipeline.run(word) for word in words]
        else:
            return words

    def similarity(self, word1, word2):
        """두 단어간의 유사도를 계산함"""
        return self.fasttext.similarity(self.word_to_jamo_pipeline.run(word1), self.word_to_jamo_pipeline.run(word2))

    def get_word_vector(self, word):
        return self.fasttext.wv.word_vec(self.word_to_jamo_pipeline.run(word))
