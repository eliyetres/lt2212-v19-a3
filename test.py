import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.


# python test.py -N 4 output300_100.test.csv 300_100_model

def load_data(filename):
    """ Loads training data from a file. """
    data = pd.DataFrame(pd.read_csv(filename))
    labels = data.iloc[:, 0]  # Use lables as classes
    vectors = data.iloc[:, 1:-1]  # The rest is the data
    return vectors, labels


def load_model(filename):
    """ Loads the trained data model """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def test_data(data, model):
    """  Takes a training data object and a trained model and """
    vectors, labels = data  # X,y
    trained_class = model.classes_
    # The softmax function is used to find the predicted probability of each class
    log = model.predict_log_proba(vectors)
    prob = model.predict_proba(vectors)
    # Entropy
    total_entropy = []
    for p, l in zip(log, prob):
        entr = 0
        for pp, ll in zip(p, l):
            entr += (-1*pp*ll)
        total_entropy.append(entr)
    enl = len(total_entropy)
    total_entropy = (sum(total_entropy))/enl
    # Perplexity
    total_perplexity = 2**total_entropy
    # Accuracy
    test_predictions = model.predict(vectors)
    total_accuracy = model.score(vectors, labels)
    print("Predicted classes: ", trained_class)
    print("Test data predictions", test_predictions)
    return total_accuracy, total_perplexity


parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                    default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
data = load_data(args.datafile)
print("Loading model from file {}.".format(args.modelfile))
model = load_model(args.modelfile)
a, p = test_data(data, model)


print("Testing {}-gram model.".format(args.ngram))

print("Accuracy is {}".format(a))
print("Perplexity is {}".format(p))
