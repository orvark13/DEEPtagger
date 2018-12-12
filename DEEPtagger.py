import random
import numpy as np
from collections import Counter, defaultdict
from itertools import count
from time import time
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


class Meta:
    def __init__(self):
        self.c_dim = 32
        self.n_hidden = 32
        self.lstm_char_input_dim = 20
        self.lstm_word_input_dim = 128
        self.lstm_tags_input_dim = 30
        self.lstm_char_output_dim = 64
        self.lstm_word_output_dim = 64


class DEEPTagger():
    def __init__(self, model=None, meta=None):
        self.model = dy.Model()
        self.meta = meta

        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.meta.nwords, self.meta.lstm_word_input_dim))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.meta.nchars, self.meta.lstm_char_input_dim))
        #self.p_t1 = self.model.add_lookup_parameters((self.meta.ntags, self.meta.lstm_tags_input_dim))

        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.pH = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_output_dim * 2))  # vocab-size, input-dim
        self.pO = self.model.add_parameters((self.meta.ntags, self.meta.n_hidden))  # vocab-size, hidden-dimnwords, self.meta.lstm_word_dim))

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_word_input_dim * 2, self.meta.lstm_word_output_dim, self.model) # layers, input-dim, output-dim
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_word_input_dim * 2, self.meta.lstm_word_output_dim, self.model)

        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(1, self.meta.lstm_char_input_dim, self.meta.lstm_char_output_dim, self.model)
        self.cBwdRNN = dy.LSTMBuilder(1, self.meta.lstm_char_input_dim, self.meta.lstm_char_output_dim, self.model)

    def dynamic_rep(self, w, cf_init, cb_init):
        if meta.wc[w] >= HP_WEMB_MIN_FREQ:
            w_index = meta.vw.w2i[w]
            return self.WORDS_LOOKUP[w_index]
        else:
            pad_char = meta.vc.w2i[UNKNOWN_CHAR]
            char_ids = [pad_char] + [meta.vc.w2i[c] for c in w] + [pad_char]
            char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
            fw_exps = cf_init.transduce(char_embs)
            bw_exps = cb_init.transduce(reversed(char_embs))
            return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def char_rep(self, w, cf_init, cb_init):
        pad_char = meta.vc.w2i[UNKNOWN_CHAR]
        char_ids = [pad_char] + [meta.vc.w2i[c] for c in w] + [pad_char]
        char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def word_rep(self, w):
        if meta.wc[w] == 0:
            return dy.zeros(self.meta.lstm_word_input_dim, batch_size=1)
        w_index = meta.vw.w2i[w]
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

        # Get the vectors, a 128-dim vector expression for each word.
        if DYNAMIC_TAGGING:
            wembs = [self.dynamic_rep(w, cf_init, cb_init) for w in words]
        else:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init) for w in words]
        wembs = [dy.noise(we, HP_EMB_NOISE) for we in wembs]

        # Feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))

        # biLSTM states
        bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

        # Feed each biLSTM state to an MLP
        exps = []
        for x in bi_exps:
            r_t = self.pO * (dy.tanh(self.pH * x))
            exps.append(r_t)
        return exps

    def sent_loss(self, sent):
        words, tags = map(list, zip(*sent))
        vecs = self.build_tagging_graph(words)
        errs = []
        for v, t in zip(vecs, tags):
            tid = meta.vt.w2i[t]
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
            tags.append(meta.vt.i2w[tag])
        return zip(words, tags)


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


def write_to_sheets():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)

    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    sheet = client.open("DEEPtagger: Experiments and results").sheet1

    # Extract and print all of the values
    # list_of_hashes = sheet.get_all_records()
    # print(list_of_hashes)

    row = [logging_dict['epoch'], logging_dict['word_acc'], logging_dict['sent_acc'], logging_dict['known_acc'],
           logging_dict['unknown_acc'], logging_dict['ifd_set'], logging_dict['optimization'], logging_dict['min_freq'],
           logging_dict['noise'], logging_dict['learning_rate'], logging_dict['learning_rate_max'],
           logging_dict['learning_rate_min'], logging_dict['dynamic'], logging_dict['memory'], logging_dict['random_seed'],
           logging_dict['dropout'], logging_dict['seconds'], logging_dict['final']]
    sheet.insert_row(row, 2)



def evaluate_on_dev():
    good = total = good_sent = total_sent = unk_good = unk_total = 0.0
    for sent in dev:
        words, golds = map(list, zip(*sent))
        tags = [t for _, t in tagger.tag_sent(words)]
        if tags == golds: good_sent += 1
        total_sent += 1
        for go, gu, w in zip(golds, tags, words):
            if go == gu:
                good += 1
                if meta.wc[w] == 0: unk_good += 1
            total += 1
            if meta.wc[w] == 0: unk_total += 1
    print("OOV", unk_total, ", correct ", unk_good)
    return good/total, good_sent/total_sent, (good-unk_good)/(total-unk_total), unk_good/unk_total


def train_tagger(train):
    num_tagged = cum_loss = 0
    start_time = time()
    for ITER in range(HP_NUM_EPOCHS):
        epoch_start_time = time()
        random.shuffle(train)
        for i, sent in enumerate(train, 1):
            # Report state
            if i > 0 and i % 5000 == 0:
                trainer.status()
                print("AVG LOSS: {:f}".format(cum_loss / num_tagged))
                cum_loss = num_tagged = 0

            # Training
            loss_exp = tagger.sent_loss(sent)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(sent)
            loss_exp.backward()
            trainer.update()

            # Evaluate
            if i == len(train) - 1:
                eval_values = evaluate_on_dev()
                seconds = str(time() - epoch_start_time)
                print("EVAL: tags {:.3%} sent {:.3%} knw {:.3%} unk {:.3%}".format(*eval_values))
                logging_dict.update({'epoch': ITER+1, 'word_acc': eval_values[0], 'sent_acc': eval_values[1],
                                     'known_acc': eval_values[2], 'unknown_acc': eval_values[3], 'seconds': seconds})
                if ITER+1 == HP_NUM_EPOCHS:
                    logging_dict['final':'X']
                write_to_sheets()

        print("------- EPOCH {:^2} DONE -------".format(ITER))
        now_time = time()
        print("Seconds since start {:f}, epoch {:f}".format(now_time - start_time, now_time - epoch_start_time))

    print("HP epochs={} wemb_min={} emb_noise={} ".format(HP_NUM_EPOCHS, HP_WEMB_MIN_FREQ, HP_EMB_NOISE))


def set_trainer(optimization, model):
    if optimization == 'MomentumSGD':
        return dy.MomentumSGDTrainer(model, learning_rate=HP_LEARNING_RATE)
    if optimization == 'CyclicalSGD':
        return dy.CyclicalSGDTrainer(model, learning_rate_max=HP_LEARNING_RATE_MAX, learning_rate_min=HP_LEARNING_RATE_MIN)
    if optimization == 'SimpleSGD':
        return dy.SimpleSGDTrainer(model, learning_rate=HP_LEARNING_RATE)
    if optimization == 'Adam':
        return dy.AdamTrainer(model)
    if optimization == 'RMSProp':
        return dy.RMSPropTrainer(model)

meta = Meta()
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
    parser.add_argument('--dynamic', '-dyn', help="Tag dynamically", type=bool, default=False)
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

    logging_dict = {'ifd_set': IFD_SET_NUM, 'min_freq': '', 'noise': HP_EMB_NOISE, 'learning_rate': '',
                    'learning_rate_max': '', 'learning_rate_min': '', 'dropout': HP_DROPOUT,
                    'optimization': OPTIMIZATION_MODEL, 'dynamic': DYNAMIC_TAGGING, 'final': '', 'memory': args.mem,
                    'random_seed': args.random_seed}

    if DYNAMIC_TAGGING:
        logging_dict['min_freq'] = HP_WEMB_MIN_FREQ
    if OPTIMIZATION_MODEL == 'CyclicalSGD':
        logging_dict['learning_rate_max'] = HP_LEARNING_RATE_MAX
        logging_dict['learning_rate_min'] = HP_LEARNING_RATE_MIN
    if OPTIMIZATION_MODEL in ['MomentumSGD','SimpleSGD']:
        logging_dict['learning_rate'] = HP_LEARNING_RATE

    UNKNOWN_WORD = "_UNK_"
    UNKNOWN_CHAR = "<*>"
    IFD_FOLDER = './IFD/'

    train_file = IFD_FOLDER + format(IFD_SET_NUM, '02') + "TM.txt"  # FIX Have to download if missing
    test_file = IFD_FOLDER + format(IFD_SET_NUM, '02') + "PM.txt"  # FIX Have to download if missing

    words = []
    tags = []
    chars = set()
    meta.wc = Counter()

    train = list(read(train_file))
    dev = list(read(test_file))

    for sent in train:
        for w, p in sent:
            words.append(w)
            tags.append(p)
            chars.update(w)
            meta.wc[w] += 1

    for sent in dev: # Also account for chars in dev, so there are no unknown characters.
        for w, _ in sent:
            chars.update(w)

    words.append(UNKNOWN_WORD)
    chars.add(UNKNOWN_CHAR)

    meta.vw = Vocab.from_corpus([words])
    meta.vt = Vocab.from_corpus([tags])
    meta.vc = Vocab.from_corpus([chars])
    UNK = meta.vw.w2i[UNKNOWN_WORD]

    meta.nwords = meta.vw.size()
    meta.ntags = meta.vt.size()
    meta.nchars = meta.vc.size()

    tagger = DEEPTagger(meta=meta)
    trainer = set_trainer(OPTIMIZATION_MODEL, tagger.model)

    train_tagger(train)

    print("Sentences, train=", len(train), ", validation=", len(dev))

    #print(tagger.tag_sent("Markarinn er tilbúinn í slaginn!".split()))
