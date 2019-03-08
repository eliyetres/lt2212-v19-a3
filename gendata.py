import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# gendata.py -- Don't forget to put a reasonable amount code comments
import re
from nltk import ngrams
# in so that we better understand what you're doing when we grade!


def readfile(filename, startline=0, endline=None):
    """ Opens and reads a file with words and their POS-tags.
    Returns a list of words, starting and ending at n selected lines."""
    text_tag = []
    only_words = []
    # Part for words only
    wordlist = []
    with open(filename) as fp:
        for line in fp:

            w = re.sub(r'\n', '', line)
            w = re.sub(r'((?<=[a-z1-9])\/((\+?[A-Z]+)\$?-?)+|)', '', w)
            w = re.sub(r'((?<=([a-z]\'))\/([A-Z]+))', '', w)
            w = re.sub(r'((?<=[.,\!?-\`])\/([.,\!?-\`]))', '', w)
            w = re.sub(r' {1,}', ' ', w).split(" ")
            only_words.append(w)
            # print(line)
            # print(w)
    if startline > 0 and endline is not None:

        only_words = only_words[startline:endline]
        for lst in only_words:
            for word in lst:
                wordlist.append(word)
            return wordlist
    if startline > 0:
        only_words = only_words[startline:]
    if endline is not None:
        only_words = only_words[endline:]
    # ((?<=[a-z])\/([A-Z]+-?)+)|((?<=[.,\'])\/[.,\'])
    # This part is for the words with tags
    #taglist = text.split(" ")
    """ for pair in taglist:
    p = tuple(pair.split("/"))        
    text_tag.append(p) """

    for lst in only_words:
        for word in lst:
            wordlist.append(word)
    return wordlist


#words = readfile("brown_rga.txt", 1, 5)

def generate_vocab(wordlist):
    """ Takes a list of all  words in the vocabulary.
    Returns an enumerated list of tuples for each word and its corresponding ID """    
    vocab = list(enumerate(set(wordlist+["<s>"] ))) # Adding start tag to end of vocabulary
    print("Example of word and it's ID: ", vocab[:3], vocab[:-3])
    return vocab

#vocab = generate_vocab(words)


def create_ngram(words, n=3):
    """ Takes a list of word form the vocabulary.
    Returns the selected number of n-grams as a list of tuples, default is 3 """
    n_grams = ngrams(words, n, pad_left=True,
                     left_pad_symbol='<s>', pad_right=False) # Padding n-grams with start tag
    return n_grams


#n_grams = list(create_ngram(words, 3))
# print(n_grams)

def one_hot(vocab, n_grams):
    """ Takes a numpy array of n-grams and the full vocabulary.
    Matches the words' ID with the words in the n-gram and adds the ending word of the n-gram as a label.
    Returns a numpy array of the n-grams' one-hot encodings and labels"""
   
    ngram_len = len(n_grams)
    print("Number of words in the vocabulary: ", len(vocab))
    print("Number of n-grams: ", ngram_len)
    vocab_ids = [id[0] for id in vocab] # word ids
    arr = np.eye(ngram_len, dtype=int)[vocab_ids] # Fill an array with 0s, add 1
    print(arr)
    v_len = ngram_len* (args.ngram-1) # Length of 
    one_hot_vectors = np.empty((0,v_len),int)
    class_value_labels = [] 
    for gram in n_grams:       
        #print("N-gram: ", gram)  
        #print("Label", gram[-1])      
        class_value_labels.append(gram[-1]) # Use last word of n-gram as class value label
        ngram_vector = np.array([],dtype=int)       
        for word in gram[:-1]:            
            #print("Word in ngram: ", word)
            word_index = [item for item in vocab if word in item]
            word_arr = arr[word_index[0][0]] # Corresponding index
            ngram_vector = np.append(ngram_vector, word_arr)      
        #ngram_vector.flatten()
        #print("N-gram vector: ", ngram_vector)
        one_hot_vectors = np.vstack([one_hot_vectors, ngram_vector])
    print("Concat vector: ", one_hot_vectors)
    vector_obj = pd.DataFrame(data =one_hot_vectors)
    vector_obj['label'] = class_value_labels # Add word labels to last column
    #print(vector_obj)
    return vector_obj

#one_hot(vocab, n_grams)

def print_to_file(data, filename):
    """Takes a data object and prints it to a CVS file. """
    if filename[-3:] == "csv":
        print("Creating csv file.")
        pd.DataFrame(data).to_csv(filename, mode='w') # 
    else:
        np.savetxt(filename, data)


# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.
parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                    default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()


print("Loading data from file {}.".format(args.inputfile))

print("Starting from line {}.".format(args.startline))

if args.endline:
    print("Ending at line {}.".format(args.endline))
    words = readfile(args.inputfile, args.startline, args.endline)
else:
    print("Ending at last line of file.")
    words = readfile(args.inputfile, args.startline, args.endline)

vocab = generate_vocab(words)
n_grams = list(create_ngram(words, args.ngram))
one_data = one_hot(vocab, n_grams)

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))
print_to_file(one_data, args.outputfile)

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
