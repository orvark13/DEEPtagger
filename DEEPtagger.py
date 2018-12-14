import random
import numpy as np
from collections import Counter, defaultdict
from itertools import count

import dynet_config
dynet_config.set(mem=1024, random_seed=42)
random.seed(42)
import dynet as dy

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).__next__)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).__next__)
        for sentence in corpus:
            [w2i[word] for word in sentence]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())


class Dimensions:
    def __init__(self):
        self.hidden = 32
        self.hidden_input = 100
        self.char_input = 20
        self.word_input = 100 #128
        self.tags_input = 30
        self.char_output = 50 #64
        self.word_output = 50 #64
        self.word_lookup = 100 #128
        self.char_lookup = 20
class ConcatDimensions:
    def __init__(self):
        self.hidden = 32
        self.hidden_input = 128
        self.char_input = 20
        self.word_input = 256
        self.tags_input = 30
        self.char_output = 64
        self.word_output = 64
        self.word_lookup = 128
        self.char_lookup = 20


class DEEPTagger():
    START_OF_WORD = "<w>"
    END_OF_WORD = "</w>"

    def __init__(self, hyperparams):
        self.hp = hyperparams

        self.model = dy.Model()
        self.trainer = None
        self.word_frequency = Counter()

        self.set_trainer(self.hp.optimization)

        if self.hp.dynamic:
            self.dim = Dimensions()
        else:
            self.dim = ConcatDimensions()


    def create_network(self):
        assert self.vw.size(), "Need to build the vocabulary (build_vocab) before creating the network."

        if self.hp.pre_trained_embeddings is not None:  # Use pretrained embeddings
            self.dim.word_lookup = self.word_embeddings.shape[1]
            self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.vw.size(), self.dim.word_lookup))
            self.WORDS_LOOKUP.init_from_array(self.word_embeddings)
        else:
            self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.vw.size(), self.dim.word_lookup))

        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.vw.size(), self.dim.word_lookup))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.vc.size(), self.dim.char_lookup))
        #self.p_t1 = self.model.add_lookup_parameters((self.vocab.ntags, self.dim.tags_input))

        # MLP on top of biLSTM outputs, word/char out -> hidden -> num tags
        self.pH = self.model.add_parameters((self.dim.hidden, self.dim.hidden_input))  # hidden-dim, hidden-input-dim
        self.pO = self.model.add_parameters((self.vt.size(), self.dim.hidden))  # vocab-size, hidden-dim

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.dim.word_lookup+self.dim.char_output*2, self.dim.word_output, self.model) # layers, input-dim, output-dim
        self.bwdRNN = dy.LSTMBuilder(1, self.dim.word_lookup+self.dim.char_output*2, self.dim.word_output, self.model)

        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)
        self.cBwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)


    def set_trainer(self, optimization):
        if optimization == 'MomentumSGD':
            self.trainer = dy.MomentumSGDTrainer(self.model, learning_rate=self.hp.learning_rate)
        if optimization == 'CyclicalSGD':
            self.trainer = dy.CyclicalSGDTrainer(self.model, learning_rate_max=self.hp.learning_rate_max, learning_rate_min=self.hp.learning_rate_min)
        if optimization == 'Adam':
            self.trainer = dy.AdamTrainer(self.model)
        if optimization == 'RMSProp':
            self.trainer = dy.RMSPropTrainer(self.model)
        else: # 'SimpleSGD'
            self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=self.hp.learning_rate)

    def dynamic_rep(self, w, cf_init, cb_init):
        if self.word_frequency[w] >= self.hp.words_min_freq:
            w_index = self.vw.w2i[w]
            return self.WORDS_LOOKUP[w_index]
        else:
            return self.char_rep(w, cf_init, cb_init)

    def char_rep(self, w, cf_init, cb_init):
        char_ids = [self.vc.w2i[DEEPTagger.START_OF_WORD]] + [self.vc.w2i[c] if c in self.vc.w2i else -1 for c in w] + [self.vc.w2i[DEEPTagger.END_OF_WORD]]
        char_embs = [self.CHARS_LOOKUP[cid] if cid != -1 else dy.zeros(self.dim.char_lookup) for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def word_rep(self, w):
        if self.word_frequency[w] == 0:
            return dy.zeros(self.dim.word_lookup)
        w_index = self.vw.w2i[w]
        return self.WORDS_LOOKUP[w_index]

    def word_and_char_rep(self, w, cf_init, cb_init):
        wembs = self.word_rep(w)
        cembs = self.char_rep(w, cf_init, cb_init)
        return dy.concatenate([wembs, cembs])

    def build_tagging_graph(self, words):
        dy.renew_cg()

        # Initialize the LSTMs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()

        # Get the word vectors, a 128-dim vector expression for each word.
        if self.hp.dynamic:
            wembs = [self.dynamic_rep(w, cf_init, cb_init) for w in words]
        else:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init) for w in words]

        if self.hp.noise > 0:
            wembs = [dy.noise(we, self.hp.noise) for we in wembs]

        # Feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))

        # biLSTM states
        bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

        # Feed each biLSTM state to an MLP
        return [self.pO * (dy.tanh(self.pH * x)) for x in bi_exps]

    def sent_loss(self, sent):
        words, tags = map(list, zip(*sent))
        vecs = self.build_tagging_graph(words)
        errs = []
        for v, t in zip(vecs, tags):
            tid = self.vt.w2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self, words):
        vecs = self.build_tagging_graph(words)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.vt.i2w[tag])
        return zip(words, tags)

    def update_trainer(self):
        self.trainer.update()
        #self.trainer.status()

    def train(self, epochs, training_data):
        self.build_vocab(training_data)
        if self.hp.pre_trained_embeddings is not None:
            self.add_pre_trained_embeddings(self.hp.pre_trained_embeddings)
        self.create_network()

        for ITER in range(epochs):
            random.shuffle(training_data)
            for i, sent in enumerate(training_data, 1):
                loss_exp = self.sent_loss(sent)
                loss_exp.backward()
                self.update_trainer()

    def build_vocab(self, training_data):
        words = []
        tags = []
        chars = set()

        for sent in training_data:
            for w, p in sent:
                words.append(w)
                tags.append(p)
                chars.update(w)
                self.word_frequency[w] += 1

        chars.add(DEEPTagger.START_OF_WORD)
        chars.add(DEEPTagger.END_OF_WORD)

        self.vw = Vocab.from_corpus([words])
        self.vt = Vocab.from_corpus([tags])
        self.vc = Vocab.from_corpus([chars])

    def add_pre_trained_embeddings(self, emb_file):
        igc_vocab = {}
        igc_vectors = []
        with open(emb_file) as f:
            f.readline()
            for i, line in enumerate(f):
                igc_fields = line.strip().split(" ")
                igc_vocab[igc_fields[0]] = i
                igc_vectors.append(list(map(float, igc_fields[1:])))

        self.word_embeddings = np.zeros((len(self.vw.w2i.keys()), len(igc_vectors[0])))

        for training_word in self.vw.w2i.keys():
            try:
                self.word_embeddings[self.vw.w2i[training_word]] = self.igc_vectors[self.igc_vocab[training_word]]
            except:
                pass
        self.word_embeddings = self.word_embeddings * int(self.hp.scale_embeddings)
