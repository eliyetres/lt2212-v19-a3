import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# train.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.


def load_data(filename):
    """ Loads training data from a file. """
    df = pd.DataFrame(pd.read_csv(filename, index_col=-0))
    #print(df)
    return df


def traindata(data):
    """ Takes a pandas data object, using LogisticRegression to train on the data and return ?? """
    #print(data)
    # rows and columns
    labels = data.iloc[:, -0:-0] # Add lables as classes
    #labels = data.iloc[:, -1:-1] # Add lables as classes
    print("labels: ", labels)
    vectors = data.iloc[:,0:-2] # rest are vectors
    #print("vectors", vectors)
    #trained_data = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial' ).fit(labels, vectors) 
    print_to_file(labels, "labels.csv")
    print_to_file(vectors, "vectors.csv")
    #print("Trained data")
    #print(trained_data)
    return data


def print_to_file(data, filename):
    """Takes a data object and prints it to a CVS file. """
    if filename[-3:] == "csv":
        print("Creating csv file.")
        pd.DataFrame(data).to_csv(filename, mode='w') # 
    else:
        np.savetxt(filename, data)

parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features.")
parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
data = load_data(args.datafile)
trained_data = traindata(data)
print("Training {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.modelfile))
print_to_file(trained_data, args.modelfile)

# YOU WILL HAVE TO FIGURE OUT SOME WAY TO INTERPRET THE FEATURES YOU CREATED.
# IT COULD INCLUDE CREATING AN EXTRA COMMAND-LINE ARGUMENT OR CLEVER COLUMN
# NAMES OR OTHER TRICKS. UP TO YOU.
