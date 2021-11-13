import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer  #import stemmer
stemmer = PorterStemmer()   # create a stemmer

def tokenize(sentence):
    # Tokenization of a sentence

    return nltk.word_tokenize(sentence)

def stem(word):
    # convert the words in lower case and stemming them
    
    return stemmer.stem(word.lower())

def bag_of_words(tokenizedSentence, everyWords):

    # First stemming the tokenized Sentence
    stemmedSentence = [stem(word) for word in tokenizedSentence]

    # Take an array of zeros
    bagOfWords = np.zeros(len(everyWords), dtype=np.float32)
    for index, word in enumerate(everyWords):
        if word in stemmedSentence:
            bagOfWords[index] = 1;
        
    return bagOfWords
