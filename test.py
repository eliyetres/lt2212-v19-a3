import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

def load_data(filename):
    """ Loads training data from a file. """
    data = pd.DataFrame(pd.read_csv(filename, index_col=-0))
    labels = data.iloc[:,-1] # Use lables as classes    
    vectors = data.iloc[:,0:-2] # The rest are the data
    return vectors, labels

def load_model(filename):
    """ Loads the trained data model """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def test_data(data, model):
    """  attribute classes_ and the predict_log_proba and predict_proba methods """ 
    trained_class = model.classes_ 
   
    predicted_log = model.predict_log_proba  # The softmax function is used to find the predicted probability of each class.
    predict_prob = model.predict_proba
    print("Trained classes: ", trained_class)
    print(len(trained_class))
    trl = len(trained_class)
    print(len(data[1][:trl]))
    print("Log probability: ", predicted_log)
    print("Probability: ", predict_prob)

    # Accuracy
    acc = accuracy_score(data[1][:trl], trained_class)
    #print("Mean accuracy: ", acc)

    return acc

parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
data = load_data(args.datafile)
print("Loading model from file {}.".format(args.modelfile))
model  = load_model(args.modelfile)
acc = test_data(data, model)


print("Testing {}-gram model.".format(args.ngram))

print("Accuracy is {}%".format(acc))
print("Perplexity is...")
