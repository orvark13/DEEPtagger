#! /usr/bin/env python3

import random
import numpy as np
from collections import Counter, defaultdict
from itertools import count, cycle
from time import time
from datetime import datetime
from pathlib import Path
import argparse
import gspread
from oauth2client.service_account import ServiceAccountCredentials


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
        self.word_input = 100
        self.tags_input = 30
        self.char_output = 50
        self.word_output = 50
        self.word_lookup = 100
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
    def __init__(self):
        self.model = dy.Model()
        self.trainer = None
        self.word_frequency = Counter()

        if DYNAMIC_TAGGING:
            self.dim = Dimensions()
        else:
            self.dim = ConcatDimensions()


    def create_network(self):
        assert self.vw.size(), "Need to build the vocabulary (build_vocab) before creating the network."

        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.vw.size(), self.dim.word_lookup))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.vc.size(), self.dim.char_lookup))
        #self.p_t1 = self.model.add_lookup_parameters((self.vocab.ntags, self.dim.tags_input))

        # MLP on top of biLSTM outputs, word/char out -> hidden -> num tags
        self.pH = self.model.add_parameters((self.dim.hidden, self.dim.hidden_input))  # hidden-dim, hidden-input-dim
        self.pO = self.model.add_parameters((self.vt.size(), self.dim.hidden))  # vocab-size, hidden-dim

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.dim.word_input, self.dim.word_output, self.model) # layers, input-dim, output-dim
        self.bwdRNN = dy.LSTMBuilder(1, self.dim.word_input, self.dim.word_output, self.model)

        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)
        self.cBwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)


    def set_trainer(self, optimization):
        if optimization == 'MomentumSGD':
            self.trainer = dy.MomentumSGDTrainer(self.model, learning_rate=HP_LEARNING_RATE)
        if optimization == 'CyclicalSGD':
            self.trainer = dy.CyclicalSGDTrainer(self.model, learning_rate_max=HP_LEARNING_RATE_MAX, learning_rate_min=HP_LEARNING_RATE_MIN)
        if optimization == 'Adam':
            self.trainer = dy.AdamTrainer(self.model)
        if optimization == 'RMSProp':
            self.trainer = dy.RMSPropTrainer(self.model)
        else: # 'SimpleSGD'
            self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=HP_LEARNING_RATE)

    def dynamic_rep(self, w, cf_init, cb_init):
        if self.word_frequency[w] >= HP_WEMB_MIN_FREQ:
            w_index = self.vw.w2i[w]
            return self.WORDS_LOOKUP[w_index]
        else:
            return self.char_rep(w,  cf_init, cb_init)

    def char_rep(self, w, cf_init, cb_init):
        char_ids = [self.vc.w2i[START_OF_WORD]] + [self.vc.w2i[c] if self.vc.w2i[c] else -1 for c in w] + [self.vc.w2i[END_OF_WORD]]
        char_embs = [self.CHARS_LOOKUP[cid] if cid != -1 else dy.zeros(self.dim.char_input) for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def word_rep(self, w):
        if self.word_frequency[w] == 0:
            return dy.zeros(self.dim.word_input)
        w_index = self.vw.w2i[w]
        return self.WORDS_LOOKUP[w_index]

    def word_and_char_rep(self, w, cf_init, cb_init):
        wembs = self.word_rep(w)
        cembs = self.char_rep(w, cf_init, cb_init)
        x = dy.concatenate([wembs, cembs])
        return x

    def build_tagging_graph(self, words):
        dy.renew_cg()

        # Initialize the LSTMs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()

        # Get the word vectors, a 128-dim vector expression for each word.
        if DYNAMIC_TAGGING:
            wembs = [self.dynamic_rep(w, cf_init, cb_init) for w in words]
        else:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init) for w in words]

        if HP_EMB_NOISE > 0:
            wembs = [dy.noise(we, HP_EMB_NOISE) for we in wembs]

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

        chars.add(START_OF_WORD)
        chars.add(END_OF_WORD)

        self.vw = Vocab.from_corpus([words])
        self.vt = Vocab.from_corpus([tags])
        self.vc = Vocab.from_corpus([chars])


######### HELPERS FOR RUNNING IN CONSOLE #########

def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w, p = line
            sent.append((w, p))


spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')

def update_progress_notice(i, epoch, start_time, epoch_start_time, avg_loss, evaluation = None):
    now_time = time()
    print(" ",
        next(spinner),
        "{:>2}/{}".format(epoch, HP_NUM_EPOCHS),
        ("  {:>4}/{:<5}".format(int(now_time - start_time), str(int(now_time - epoch_start_time)) + 's') if i % 100 == 0 or evaluation else ""),
        ("  AVG LOSS: {:.3}".format(avg_loss) if i % 1000 == 0 or evaluation else ""),
        ("  EVAL: tags {:.3%} sent {:.3%} knw {:.3%} unk {:.3%}".format(*evaluation) if evaluation else ""),
        end='\r'
    )


def send_data_to_google_sheet(epoch, evaluation):
    secret_file = Path(GOOGLE_SHEETS_CREDENTIAL_FILE)
    if secret_file.is_file():
        word_acc, sent_acc, known_acc, unknown_acc = evaluation

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
        client = gspread.authorize(creds)

        # Open a Google Sheet, by name
        sheet = client.open("DEEPtagger: Experiments and results").sheet1

        row = [
            epoch,
            word_acc,
            sent_acc,
            known_acc,
            unknown_acc,
            IFD_SET_NUM, # IDF set
            OPTIMIZATION_MODEL,
            (HP_WEMB_MIN_FREQ if DYNAMIC_TAGGING else ""),
            HP_EMB_NOISE,
            (HP_LEARNING_RATE if OPTIMIZATION_MODEL in ['MomentumSGD','SimpleSGD'] else ""), # learning rate
            (HP_LEARNING_RATE_MAX if OPTIMIZATION_MODEL == 'CyclicalSGD' else ""), # learning rate max
            (HP_LEARNING_RATE_MIN if OPTIMIZATION_MODEL == 'CyclicalSGD' else ""), # learning rate min
            DYNAMIC_TAGGING,
            args.mem, # Dynet memory
            args.random_seed, # Random seed used for python and dynet
            HP_DROPOUT,
            datetime.fromtimestamp(time()).strftime("%d. %B %Y %I:%M:%S"), # timestamp
            ("X" if epoch == HP_NUM_EPOCHS else "") # is final epoch
        ]

        sheet.insert_row(row, 2)


def evaluate_tagging(tagger, test_data):
    good = total = good_sent = total_sent = unk_good = unk_total = 0.0
    for sent in test_data:
        words, golds = map(list, zip(*sent))
        tags = [t for _, t in tagger.tag_sent(words)]
        if tags == golds: good_sent += 1
        total_sent += 1
        for go, gu, w in zip(golds, tags, words):
            if go == gu:
                good += 1
                if tagger.word_frequency[w] == 0: unk_good += 1
            total += 1
            if tagger.word_frequency[w] == 0: unk_total += 1
    #print("OOV", unk_total, ", correct ", unk_good)
    return good/total, good_sent/total_sent, (good-unk_good)/(total-unk_total), unk_good/unk_total


def train_and_evaluate_tagger(tagger, training_data, test_data):
    '''
    Train the tagger, report progress to console and send to Google Sheets.
    '''
    tagger.build_vocab(training_data)
    tagger.create_network()

    start_time = time()
    for ITER in range(HP_NUM_EPOCHS):
        cum_loss = num_tagged = 0
        epoch_start_time = time()
        random.shuffle(training_data)
        for i, sent in enumerate(training_data, 1):
            # Training
            loss_exp = tagger.sent_loss(sent)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(sent)
            loss_exp.backward()
            tagger.update_trainer()

            if i % 10 == 0:
                update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged)

        # Evaluate
        evaluation = evaluate_tagging(tagger, test_data)
        update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged, evaluation)
        send_data_to_google_sheet(ITER + 1, evaluation)

    # Show hyperparameters used when we are done
    print("\nHP opt={} dynamic={} wemb_min_freq={} epochs={} wemb_min={} emb_noise={} ".format(OPTIMIZATION_MODEL, DYNAMIC_TAGGING, HP_WEMB_MIN_FREQ, HP_NUM_EPOCHS, HP_WEMB_MIN_FREQ, HP_EMB_NOISE)) # TODO add more HP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mem', default=1024)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--ifd_set', '-i', help="select which IFD set to use (1-10)", type=int, default=2)
    parser.add_argument('--epochs', '-e', help="How many epochs? (20 is default)", type=int, default=20)
    parser.add_argument('--words_min_freq', '-wmf', help="Minimum frequency of words, else use char embeddings.",
                        type=int, default=3)
    parser.add_argument('--noise', '-n', help="Noise in embeddings", type=float, default=0.1)
    parser.add_argument('--optimization', '-o', help="Which optimization algorithm",
                        choices=['SimpleSGD', 'MomentumSGD', 'CyclicalSGD', 'Adam', 'RMSProp'], default='CyclicalSGD')
    parser.add_argument('--learning_rate', '-l', help="Learning rate", type=float, default=0.1)
    parser.add_argument('--learning_rate_max', '-l_max', help="Learning rate max for Cyclical SGD", type=float,
                        default=0.1)
    parser.add_argument('--learning_rate_min', '-l_min', help="Learning rate min for Cyclical SGD", type=float,
                        default=0.01)
    parser.add_argument('--dropout', '-d', help="Dropout rate", type=float, default=0.0)
    parser.add_argument('--dynamic', '-dyn', help="Tag dynamically", action="store_true")
    args = parser.parse_args()

    import dynet_config
    # Set random seed to have the same result each time, needs to be set before dynet is imported
    dynet_config.set(mem=args.mem, random_seed=args.random_seed)
    random.seed(args.random_seed)
    import dynet as dy

    # Hyper-parameters and constants
    IFD_SET_NUM = args.ifd_set
    HP_NUM_EPOCHS = args.epochs
    HP_WEMB_MIN_FREQ = args.words_min_freq  # Use char embeddings for words that are less frequent.
    HP_EMB_NOISE = args.noise
    HP_LEARNING_RATE = args.learning_rate
    HP_LEARNING_RATE_MAX = args.learning_rate_max
    HP_LEARNING_RATE_MIN = args.learning_rate_min
    HP_DROPOUT = args.dropout
    OPTIMIZATION_MODEL = args.optimization
    DYNAMIC_TAGGING = args.dynamic

    START_OF_WORD = "<w>"
    END_OF_WORD = "</w>"

    GOOGLE_SHEETS_CREDENTIAL_FILE = './client_secret.json'

    IFD_FOLDER = './IFD/'
    train_file = IFD_FOLDER + format(IFD_SET_NUM, '02') + "TM.txt"  # FIX Have to download if missing
    test_file = IFD_FOLDER + format(IFD_SET_NUM, '02') + "PM.txt"  # FIX Have to download if missing

    train = list(read(train_file))
    test = list(read(test_file))

    # Create a neural network tagger and train it
    tagger = DEEPTagger()
    tagger.set_trainer(OPTIMIZATION_MODEL)

    train_and_evaluate_tagger(tagger, train, test)

    #tagger.train(HP_NUM_EPOCHS, train)
    #print(tagger.tag_sent("Markarinn er tilbúinn í slaginn!".split()))
