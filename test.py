import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.
from sklearn.metrics import accuracy_score

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
    """  Takes a training data object and a trained model and """ 
    labels = data[1]
    trained_class = model.classes_   
    predicted_log = model.predict_log_proba  # The softmax function is used to find the predicted probability of each class.
    predict_prob = model.predict_proba
    print("Predicted classes: ", trained_class)
    print("True classes",labels )
    trl = len(trained_class) # Keep classes same length 
    #print("Log probability: ", predicted_log(data[0]))
    #print("Probability: ", predict_prob(data[0]))
    # Entropy / Perplexity
    total_entropy = []
    for p, l in zip(predict_prob(data[0]), predicted_log(data[0])):
        for pp, ll in zip(p,l):
            total_entropy.append(pp*ll)
    enl = len(total_entropy)
    total_entropy = (sum(total_entropy))/enl
    total_perplexity= round(2**total_entropy, 2)
    print("Total entropy :", round(-total_entropy, 2))
    print("Perplexity: ", round(2**total_entropy, 2))
    total_accuracy = accuracy_score(labels[:trl], trained_class)
    return total_accuracy, total_perplexity 

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
a, p = test_data(data, model)



print("Testing {}-gram model.".format(args.ngram))

print("Accuracy is {}".format(a))
print("Perplexity is {}".format(p))
