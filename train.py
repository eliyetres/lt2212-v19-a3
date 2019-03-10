import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# train.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.


# python train.py -N 4 output300_100.train.csv 300_100_model

def load_data(filename):
    """ Loads training data from a file. """
    #df = pd.DataFrame(pd.read_csv(filename, index_col=-0))
    df = pd.DataFrame(pd.read_csv(filename))
    return df


def train_data(data):
    """ Takes a pandas data object, using LogisticRegression to train on the data.
    Returns the trained model as a pickle object. """
    # labels = data.iloc[:, -1]  # Use lables as classes
    # vectors = data.iloc[:, 2:-1]  # The rest is the data
    labels = data.iloc[:, 0]  # Use lables as classes
    vectors = data.iloc[:, 2:-1]  # The rest is the data
    trained_data = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(vectors, labels)
    with open(args.modelfile, 'wb') as f:
        pickle.dump(trained_data, f)
    # Test model to see if data is OK.
    predictions = trained_data.predict(vectors)
    print(predictions)


parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                    default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features.")
parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
data = load_data(args.datafile)
train_data(data)
print("Training {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.modelfile))


# YOU WILL HAVE TO FIGURE OUT SOME WAY TO INTERPRET THE FEATURES YOU CREATED.
# IT COULD INCLUDE CREATING AN EXTRA COMMAND-LINE ARGUMENT OR CLEVER COLUMN
# NAMES OR OTHER TRICKS. UP TO YOU.
