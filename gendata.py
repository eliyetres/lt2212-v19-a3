import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.
import re
import random
from nltk import ngrams

# python gendata.py -N 4 -S 500 -E 900 -T 100 brown_rga.txt output


def readfile(filename, testline, startline=0, endline=None):
    """ Opens and reads a file with words and their POS-tags.
    Returns a list of words, starting and ending at n selected lines."""
    word_lines = []
    # Part for words only
    with open(filename) as fp:
        for line in fp:
            w = re.sub(r'\n', '', line)
            w = re.sub(r'((?<=[a-z1-9])\/((\+?[A-Z]+)\$?-?)+|)', '', w)
            w = re.sub(r'((?<=([a-z]\'))\/([A-Z]+))', '', w)
            w = re.sub(r'((?<=[.,\!?-\`])\/([.,\!?-\`]))', '', w)
            w = re.sub(r' {1,}', ' ', w).split(" ")
            word_lines.append(w)
            # print(line)
            # print(w)
    #print("Startline {}, endline {}".format(startline, endline))
    if startline > 0 and endline is not None:
        word_lines = word_lines[startline:endline]
    elif startline > 0 and endline is None:
        word_lines = word_lines[startline:]
    elif startline == 0 and endline is not None:
        word_lines = word_lines[endline:]

    tr = word_lines[testline:]  # Split training and test lines
    te = word_lines[:testline]
    print("Train data lines: ", len(tr))
    print("Test data lines: ", len(te))
    train_data = [item for line in tr for item in line]  # Flatten lists
    test_data = [item for line in te for item in line]

    return train_data, test_data


def generate_vocab(data_list):
    """ Takes a list of the separated training and test data words in the vocabulary.
    Returns an enumerated list of tuples for each word and its corresponding ID """
    vocab = list(enumerate(set(data_list+["<s>"]))
                 )  # Adding start tag to end of vocabulary
    print("Example of word and it's ID: ", vocab[0], vocab[-1])

    return vocab


def create_ngram(words, n=3):
    """ Takes a list of word form the vocabulary.
    Returns the selected number of n-grams as a list of tuples, default is 3. """
    grams = ngrams(words, n, pad_left=True,
                   left_pad_symbol='<s>', pad_right=False)  # Padding n-grams with start tag

    return grams


def one_hot(vocab, n_grams):
    """ Takes a numpy array of n-grams and the full vocabulary.
    Matches the words' ID with the words in the n-gram, one-hot encodes them and adds the ending word of the n-gram as a label.
    Returns a numpy array of the n-grams' one-hot encodings and labels. """
    ngram_len = len(n_grams)
    print("Number of words in the vocabulary: ", len(vocab))
    print("Number of n-grams: ", ngram_len)
    vocab_ids = [id[0] for id in vocab]  # Word ids
    print("Generating ids", vocab_ids)
    # Fill an array with 0s, add 1
    arr = np.eye(ngram_len, dtype=int)[vocab_ids]
    print("Creating arrays: ", arr)
    v_len = ngram_len * (args.ngram-1)  # Fill an empty array
    one_hot_vectors = np.empty((0, v_len), int)
    print("Creating empty vectors")
    class_value_labels = []
    for gram in n_grams:
        # Use last word of n-gram as class value label
        print("Appending n-gram")
        class_value_labels.append(gram[-1])
        ngram_vector = np.array([], dtype=int)
        for word in gram[:-1]:
            word_index = [item for item in vocab if word in item]
            word_arr = arr[word_index[0][0]]  # Corresponding index
            ngram_vector = np.append(ngram_vector, word_arr)
        one_hot_vectors = np.vstack([one_hot_vectors, ngram_vector])
    #print("Concat vector: ", one_hot_vectors)
    vector_obj = pd.DataFrame(data=one_hot_vectors)
    vector_obj['label'] = class_value_labels  # Add word labels to last column

    return vector_obj


def print_to_file(data, filename):
    """Takes a data object, a name label (test or train) and prints it to a CVS file. """
    pd.DataFrame(data).to_csv(filename, mode='w')


parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                    default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("-T", "--test", metavar="T", dest="test",
                    type=int, default=None,
                    help="Specifies how many (randomly selected) lines within the range selected by -S and -E will be designated as testing data.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

if args.test > (args.endline - args.startline):
    exit("Error: Test data sample must be smaller than selected lines sample.")
if args.startline > args.endline:
    exit("Error: Starting line must be lower than ending line.")

print("Loading data from file {}.".format(args.inputfile))

print("Starting from line {}.".format(args.startline))

print(args.endline - args.startline)

if args.endline:
    print("Ending at line {}.".format(args.endline))
    train_list, test_list = readfile(
        args.inputfile, args.test, args.startline, args.endline)
else:
    print("Ending at last line of file.")
    train_list, test_list = readfile(args.inputfile, args.test, args.startline)

print("Using {} lines as test data.".format(args.test))

# Train data
train_vocab = generate_vocab(train_list)
train_ngrams = list(create_ngram(train_list, args.ngram))
train_data_encoded = one_hot(train_vocab, train_ngrams)

# Test data
test_vocab = generate_vocab(test_list)
test_ngrams = list(create_ngram(test_list, args.ngram))
test_data_encoded = one_hot(test_vocab, test_ngrams)

print("Constructing {}-gram model.".format(args.ngram))
print("Writing train data table to {}.".format(args.outputfile+".train"))
print("Writing test data table to {}.".format(args.outputfile+".test"))

print_to_file(train_data_encoded, args.outputfile+".train.csv")  # Train data
print_to_file(test_data_encoded, args.outputfile+".test.csv")  # Test data

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
