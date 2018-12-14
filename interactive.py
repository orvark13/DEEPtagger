#! /usr/bin/env python3

import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifd_set', '-i', help="select which IFD set to use (1-10)", type=int, default=1)
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

    train_file = './IFD/' + format(args.ifd_set, '02') + "TM.txt"
    test_file = './IFD/' + format(args.ifd_set, '02') + "PM.txt"

    ifd = list(read(train_file))
    ifd += list(read(test_file))

    # Create a neural network tagger and train it
    print(" Þjálpa markara ...", end='\r')
    tagger = DEEPTagger(args)

    tagger.train(args.epochs, ifd)
    print("Markarinn er tilbúinn.")
    sent = ""
    while sent != "hætta":
        sent = input("> ")
        print(" ".join([x[0] + "/" + x[1] for x in tagger.tag_sent(sent.split())]))
