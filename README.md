# DEEPtagger
Part-of-Speech Tagging for Icelandic using a Bidirectional Long Short-Term Memory Model with combined Word and Character embeddings

[Project report](https://github.com/orvark13/DEEPtagger/raw/master/DEEPtagger-report-steinthor-orvar.pdf)

## Scripts

### Training models
A bidirectional LSTM model can be trained with the script `train.py`.

![train](https://user-images.githubusercontent.com/24220374/50121120-a8d95c80-024f-11e9-97a5-b4244b4d0bdd.png)

Running `./train.py -h` gives information on all possible parameters.

The program requires input corpora to be in the same format as the IFD-training/testing sets, available at http://www.malfong.is/index.php?lang=en&pg=ordtidnibok

Results from experiments with different models, trained on the IFD, are here: https://docs.google.com/spreadsheets/d/1YG8xYaW10x4jvsCFyHkQxdYQLIOFa6WoBUzbA4y4Gtg/edit?usp=sharing

## Fetching results during training

During long running training sessions `get_results.py` was used fetch the latest results and calculate the highest scoring epoch.

![get_results](https://user-images.githubusercontent.com/24220374/50121118-a8d95c80-024f-11e9-9064-41cca53c97c5.png)

If the parameter `-p` is added when running the script a plot is provided for the accuracy and avg. loss of all epochs finished sofar.

## Trying out the resulting PoS tagger

The script `interactive.py` trains a model and then allow the user to try out the tagger on sentences he enters. It accepts the same parameters as `train.py`.

![interactive](https://user-images.githubusercontent.com/24220374/50121119-a8d95c80-024f-11e9-902f-44364692cea7.png)
