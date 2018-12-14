#! /usr/bin/env python3

import random
from itertools import cycle
from time import time
from datetime import datetime
from pathlib import Path
import argparse
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from DEEPtagger import DEEPTagger

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
        "{:>2}/{}".format(epoch, args.epochs),
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
            args.ifd_set,
            args.optimization,
            (args.words_min_freq if args.dynamic else ""),
            args.noise,
            (args.learning_rate if args.optimization in ['MomentumSGD','SimpleSGD'] else ""),
            (args.learning_rate_max if args.optimization == 'CyclicalSGD' else ""),
            (args.learning_rate_min if args.optimization == 'CyclicalSGD' else ""),
            args.dynamic,
            1024, #args.mem, # Dynet memory allocation
            42, #args.random_seed, # Random seed used for python and Dynet
            args.dropout,
            datetime.fromtimestamp(time()).strftime("%d. %B %Y %I:%M:%S"), # timestamp
            ("X" if epoch == args.epochs else "") # is final epoch
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
    for ITER in range(args.epochs):
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
    print("\nHP opt={} dynamic={} wemb_min_freq={} epochs={} wemb_min={} emb_noise={} ".format(args.optimization, args.dynamic, args.words_min_freq, args.epochs, args.words_min_freq, args.noise))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mem', default=1024)
    #parser.add_argument('--random_seed', type=int, default=42)
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

    GOOGLE_SHEETS_CREDENTIAL_FILE = './client_secret.json'

    train_file = './IFD/' + format(args.ifd_set, '02') + "TM.txt"  # FIX Have to download if missing
    test_file = './IFD/' + format(args.ifd_set, '02') + "PM.txt"  # FIX Have to download if missing

    train = list(read(train_file))
    test = list(read(test_file))

    # Create a neural network tagger and train it
    tagger = DEEPTagger(args)

    train_and_evaluate_tagger(tagger, train, test)

    #tagger.train(HP_NUM_EPOCHS, train)
    #print(tagger.tag_sent("Markarinn er tilbúinn í slaginn!".split()))
