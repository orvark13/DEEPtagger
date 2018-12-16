#! /usr/bin/env python3

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from collections import Counter
from tabulate import tabulate
import plotly
import plotly.graph_objs as go
import argparse

GOOGLE_SHEETS_CREDENTIAL_FILE = './client_secret.json'
secret_file = Path(GOOGLE_SHEETS_CREDENTIAL_FILE)

parser = argparse.ArgumentParser()
parser.add_argument('--sheet', '-s', help="select which sheet to read", default='WC+')
parser.add_argument('--plot', '-p', help="Plot graph?",  action="store_true")
args = parser.parse_args()

if secret_file.is_file():

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)

    # Open a Google Sheet, by name
    sheet = client.open("DEEPtagger: Experiments and results").worksheet(args.sheet)
    epoch = list(map(int, filter(None, sheet.col_values(1)[1:])))
    word_acc = list(map(float, filter(None, sheet.col_values(2)[1:])))
    sent_acc = list(map(float, filter(None, sheet.col_values(3)[1:])))
    known_acc = list(map(float, filter(None, sheet.col_values(4)[1:])))
    unknown_acc = list(map(float, filter(None, sheet.col_values(5)[1:])))
    loss = list(map(float, filter(None, sheet.col_values(19)[1:])))
    ifd_set = list(map(int, filter(None, sheet.col_values(6)[1:])))
    acc_dict = {}
    sent_acc_dict = {}
    known_dict = {}
    unknown_dict = {}
    loss_dict = {}

    ctr = 0
    for i in word_acc:
        try:
            if epoch[ctr] in acc_dict.keys():
                acc_dict[epoch[ctr]] += word_acc[ctr]
                sent_acc_dict[epoch[ctr]] += sent_acc[ctr]
                known_dict[epoch[ctr]] += known_acc[ctr]
                unknown_dict[epoch[ctr]] += unknown_acc[ctr]
                loss_dict[epoch[ctr]] += loss[ctr]
            else:
                acc_dict[epoch[ctr]] = word_acc[ctr]
                sent_acc_dict[epoch[ctr]] = sent_acc[ctr]
                known_dict[epoch[ctr]] = known_acc[ctr]
                unknown_dict[epoch[ctr]] = unknown_acc[ctr]
                loss_dict[epoch[ctr]] = loss[ctr]
        except:
            pass
        ctr += 1

    epoch_ctr = Counter(epoch)

    max_val = 0.01
    best_epoch = 1
    for i in epoch_ctr.keys():
        try:
            if epoch_ctr[i] == 10 and acc_dict[i]/epoch_ctr[i] > max_val:
                max_val = acc_dict[i]/epoch_ctr[i]
                best_epoch = i
        except:
            pass

    print(tabulate([[best_epoch, acc_dict[best_epoch]/epoch_ctr[best_epoch],
                     sent_acc_dict[best_epoch]/epoch_ctr[best_epoch],
                     known_dict[best_epoch]/epoch_ctr[best_epoch],
                     unknown_dict[best_epoch]/epoch_ctr[best_epoch],
                     loss_dict[best_epoch]/epoch_ctr[best_epoch]
                    ]],
                    headers=['Best epoch','Word acc.','Sent acc.','Known','OOV','Loss']))

    if args.plot:
        y_acc = list([acc_dict[x]/epoch_ctr[x]*100.0 for x in epoch_ctr])
        y_loss = list([loss_dict[x]/epoch_ctr[x] for x in epoch_ctr])
        #y_loss = list([loss_dict[x]/epoch_ctr[x] if loss_dict[x]/epoch_ctr[x] < 0.0055 else 0.0055 for x in epoch_ctr]) # Only graph loss after epoch 40 so the curve is more distinct
        x_epoch = list(acc_dict.keys())

        trace0 = go.Scatter(
            y=y_acc,
            x=x_epoch,
            name="Tagging accuracy"
        )

        trace1 = go.Scatter(
            y=y_loss,
            x=x_epoch,
            name="Avg. loss",
            yaxis='y2'
        )

        data = [trace0, trace1]
        layout = go.Layout(
            title='Tagging accuracy and avg. loss',
            yaxis=dict(
                title='Accuracy'
            ),
            yaxis2=dict(
                title='Avg. loss',
                titlefont=dict(
                    color='rgb(148, 103, 189)'
                ),
                tickfont=dict(
                    color='rgb(148, 103, 189)'
                ),
                overlaying='y',
                type='log',
                side='right'
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, auto_open=True)
