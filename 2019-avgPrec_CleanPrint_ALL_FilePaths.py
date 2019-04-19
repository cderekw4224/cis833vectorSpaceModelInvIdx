# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:50:32 2017

@author: Derek Christensen

The Vector Space Model: Implementation

HW 2
CIS 833 Information Retrieval and Text Mining
Cornelia Caragea
Department of Computer Science
Kansas State University

"""

##################################
# IMPORTS
##################################

from __future__ import absolute_import, division, print_function
# coding: utf-8
from collections import namedtuple

__author__ = 'Derek W. Christensen'
__email__ = 'cderekw@gmail.com'
__version__ = '0.0.0'

import sys
import math
import functools as ft
# from functools import reduce


# Regular Expression
import re
import os

# In Python 2, zip returns a list, to avoid creating an unnecessary list,\
# use izip instead
# (aliased to zip can reduce code changes when you move to Python 3)
import itertools
#from itertools import izip as zip
zip = getattr(itertools, 'izip', zip)

# from collections import Counter
from collections import defaultdict

# import nltk, which is a python package for natural language processing
import nltk
# to remove stropwords
from nltk.corpus import stopwords
# FreqDist, word_tokenize
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

# Excel-like format
import pandas as pd
# to diplay max rows
pd.set_option('display.max_rows', 1000)
# to diplay max cols
pd.set_option('display.max_columns', 1000)
# to define width of cells
pd.set_option('display.max_colwidth', 1000)

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.ticker as ticker
# import ipython
#% matplotlib inline
#get_ipython().magic(u'matplotlib inline')

# #############################################################################
# FUNCTIONS
# #############################################################################

Docrow = namedtuple('Docrow',
                    ['TopXDocs', 'QNum', 'TxQ', 'RelIddxRankNum',
                     'OQD_QNum', 'OQD_DocID',
                     'Relevant', 'RelCnt', 'Recall', 'Precision'])

AvgPrec = namedtuple('AvgPrec',
                     ['p0', 'p1', 'p2', 'p3', 'p4', 'p5',
                      'p6', 'p7', 'p8', 'p9', 'p10'])

# ##########################
# Step 1: Preprocessing Functions
# ##########################


# def getStopwords(dir_path_stopwords)
# return(stopwords_from_file)


def getStopwords(dir_path_stopwords):
    files_stopwords = os.listdir(dir_path_stopwords)
    for fsw in files_stopwords:
        with open(dir_path_stopwords+'/'+os.path.basename(fsw), 'r') as swfile:
            stopwords_from_file = set(swfile.read().splitlines())
    return(stopwords_from_file)

# def getFiles(dir_path)
# return(files, file_names, file_idx, file_zip, file_dict, file_dict_enum)


def getFiles(dir_path):
    files = os.listdir(dir_path)
    file_names = os.listdir(dir_path)
    for i in range(len(file_names)):
        file_idx.append(i+1)
    file_zip = zip(file_idx, file_names)
    file_dict = dict(file_zip)
    files_dict_enum = {key: value for key, value in enumerate(file_names)}
    return(files, file_names, file_idx, file_zip, file_dict, files_dict_enum)

# def getLines(files, dir_path)
# return(review, docnum, titles, texts)


def getLines(files, dir_path):
    # tokenize the words based on white space, removes the punctuation
    strtemp = ""
    for f in files:
        with open(dir_path+'/'+os.path.basename(f), 'r') as ipfile:
            i = 0
            for line in ipfile:
                line = line.strip()
                if i == 2:
                    docnum.append(line)
                    review.append(line)
                    i += 1
                elif i == 5:
                    strtemp += line
                    strtemp += " "
                    review.append(line)
                    i += 1
                    while line != '</TITLE>':
                        for line in ipfile:
                            line = line.strip()
                            if line == '</TITLE>':
                                review.append(line)
                                i += 1
                            else:
                                strtemp += line
                                strtemp += " "
                                review.append(line)
                                i += 1
                            break
                    titles.append(strtemp)
                    strtemp = ""
                elif line == '<TEXT>':
                    review.append(line)
                    i += 1
                    while line != '</TEXT>':
                        for line in ipfile:
                            line = line.strip()
                            if line == '</TEXT>':
                                review.append(line)
                                i += 1
                            else:
                                strtemp += line
                                strtemp += " "
                                review.append(line)
                                i += 1
                            break
                    texts.append(strtemp)
                    strtemp = ""
                else:
                    review.append(line)
                    i += 1
    return(review, docnum, titles, texts)

# def getPerDocCorp(titles, texts)
# return(perDocCorp, corpus)


def getPerDocCorp(titles, texts):
    strtemp = ""
    corpustemp = ""
    for i in range(len(titles)):
        strtemp += titles[i]
        strtemp += texts[i]
        corpustemp += strtemp
        perDocCorp.append(strtemp)
        strtemp = ""
    corpus.append(corpustemp)
    return(perDocCorp, corpus)

# def getPerDocCorpClean(perDocCorp)
# return(perDocCorpClean, perDocLen, fdistPerDoc, fdistPerDocLen,
#       freq_word_PerDoc)

# for ea perDocCorp: tokenize, clean, stem, lem, stopwords, \
# shortwords, etc.


def getPerDocCorpClean(perDocCorp):
    i = 0
    for doc in perDocCorp:
        tokens = str(doc)
        # lowecases for content analytics ... we assume, for example, \
        # LOVE is sames love
        tokens = tokens.lower()
        # the dataset contains useless characters and numbers
        # Remove useless numbers and alphanumerical words
        # use regular expression ... a-zA-Z0-9 refers to all English \
        # characters (lowercase & uppercase) and numbers
        # ^a-zA-Z0-9 is opposite of a-zA-Z0-9
        tokens = re.sub("[^a-zA-Z0-9]", " ", tokens)
        # tokenization or word split
        tokens = word_tokenize(tokens)
        # Filter non-alphanumeric characters from tokens
        tokens = [word for word in tokens if word.isalpha()]
        # remove short words
        tokens = [word for word in tokens if len(word) > 2]
        # remove common words
        stoplist = stopwords.words('english')
        # if you want to remove additional words EXAMPLE
        # more = set(['much', 'even', 'time', 'story'])
        # more = set(['the'])
        # stoplist = set(stoplist) | more
        stoplist = set(stoplist) | stopwords_from_file
        stoplist = set(stoplist)
        tokens = [word for word in tokens if word not in stoplist]
        # stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # -----CLEANING COMPLETE-----
        perDocCorpClean.append(tokens)
        lenDocTokens = len(tokens)
        perDocLen.append(lenDocTokens)
        fdist = nltk.FreqDist(tokens)
        fdistPerDoc.append(fdist)
        lenDocFdist = len(fdist)
        fdistPerDocLen.append(lenDocFdist)
        # freq_word_PerDoc = []
        # prepare the results of word frequency on corpus data as a list
        freq_word = []
        # two values or columns in fdist_a
        j = 0
        for k, v in fdist.items():
            freq_word.append([k, v])
            j += 1
        # make it like an Excel worksheet
        wordlist = pd.DataFrame(freq_word)
        # pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_rows', 10)
        wordlistSorted = wordlist.sort_values(by=[1, 0],
                                              ascending=[False, True])
        freq_word_PerDoc.append(wordlistSorted)
        i += 1
    return(perDocCorpClean, perDocLen, fdistPerDoc, fdistPerDocLen,
           freq_word_PerDoc)

# def getCorpusClean(corpus)
# return(corpusClean, corpusLen, fdistCorpus, fdistCorpusLen,
#        freq_word_Corpus)

# for corpus: tokenize, clean, stem, lem, stopwords, \
# shortwords, etc.


def getCorpusClean(corpus):
    tokens = str(corpus)
    # lowecases for content analytics ... we assume, for example, \
    # LOVE is sames love
    tokens = tokens.lower()
    # the dataset contains useless characters and numbers
    # Remove useless numbers and alphanumerical words
    # use regular expression ... a-zA-Z0-9 refers to all English \
    # characters (lowercase & uppercase) and numbers
    # ^a-zA-Z0-9 is opposite of a-zA-Z0-9
    tokens = re.sub("[^a-zA-Z0-9]", " ", tokens)
    # tokenization or word split
    tokens = word_tokenize(tokens)
    # Filter non-alphanumeric characters from tokens
    tokens = [word for word in tokens if word.isalpha()]
    # remove short words
    tokens = [word for word in tokens if len(word) > 2]
    # remove common words
    stoplist = stopwords.words('english')
    # if you want to remove additional words EXAMPLE
    # more = set(['much', 'even', 'time', 'story'])
    # more = set(['the'])
    # stoplist = set(stoplist) | more
    stoplist = set(stoplist) | stopwords_from_file
    stoplist = set(stoplist)
    tokens = [word for word in tokens if word not in stoplist]
    # stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # -----CLEANING COMPLETE-----
    corpusClean.append(tokens)
    lenCorpusTokens = len(tokens)
    corpusLen.append(lenCorpusTokens)
    fdist = nltk.FreqDist(tokens)
    fdistCorpus.append(fdist)
    lenCorpusFdist = len(fdist)
    fdistCorpusLen.append(lenCorpusFdist)
    # freq_word_PerDoc = []
    # prepare the results of word frequency on corpus data as a list
    freq_word = []
    # two values or columns in fdist_a
    j = 0
    for k, v in fdist.items():
        freq_word.append([k, v])
        j += 1
    # make it like an Excel worksheet
    wordlist = pd.DataFrame(freq_word)
    # pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_rows', 10)
    wordlistSorted = wordlist.sort_values(by=[1, 0],
                                          ascending=[False, True])
    freq_word_Corpus.append(wordlistSorted)
    return(corpusClean, corpusLen, fdistCorpus, fdistCorpusLen,
           freq_word_Corpus)


# ##########################
# Step 2: Indexing Functions
# ##########################

# create Postings dictionary function

    # def getPostings(file_names, freq_word_PerDoc, perDocCorpClean)
    # return(postings)


def getPostings(file_names, freq_word_PerDoc, perDocCorpClean):
    for docid in (range(len(file_names))):
        for word in freq_word_PerDoc[docid][0]:
            postings[word][docid] = perDocCorpClean[docid].count(word)
    return(postings)

# create DF dictionary function
# getDF(file_names, freq_word_Corpus, postings)
# return(df)


def getDF(file_names, freq_word_Corpus, postings):
    for docid in (range(len(file_names))):
        for word in freq_word_Corpus[0][0]:
            df[word] = len(postings[word])
    return(df)

# ## Create Inverted Index & docVecLen Functions

# calculate IDF
# getIDF(word)
# return(idf)


def getIDF(word):
    if word in fdistCorpus[0]:
        N = (len(file_names))
        dfi = df[word]
        N_div_dfi = N / dfi
        idf = math.log(N_div_dfi, 2)
    else:
        idf = 0.0
#    print('\nidf for ', word, ' = ', idf)
#    print()
    return(idf)

# calculate weight
# getWeight(word, docid)
# return(weight)


def getWeight(word, docid):
    tf = 0
    idf = 0
    if docid in postings[word]:
        tf = postings[word][docid]
        idf = getIDF(word)
        weight = tf * idf
    else:
        tf = 0
        idf = getIDF(word)
        weight = tf * idf
    return(weight)

# create Inverted Index & docVecLen
# getDocVecLen(file_names, freq_word_Corpus)
# return(docVecLen)


def getDocVecLen(file_names, freq_word_Corpus):
    for docid in (range(len(file_names))):
        sumSquares = 0
        for word in freq_word_Corpus[0][0]:
            weight = getWeight(word, docid)
            weight_sq = weight**2
            sumSquares += weight_sq
        docVecLen[docid] = math.sqrt(sumSquares)
    return(docVecLen)


# ##########################
# Step 3: Retrival Functions
# ##########################

# def getQueries(dir_path_queries):
# return(queries_from_file)


def getQueries(dir_path_queries):
    files_queries = os.listdir(dir_path_queries)
    for fq in files_queries:
        with open(dir_path_queries+'/'+os.path.basename(fq), 'r') as qfile:
            queries_from_file = (qfile.read().splitlines())
    return(queries_from_file)

# Get lines of Input
# def getQLines(q)
# return(qReview, qDocnum, qTexts)


def getQLines(q):
    # tokenize the words based on white space, removes the punctuation
    strtemp = ""
    queryNum = 0
    qDocnum.append(queryNum)
#    print('\nqTexts in  getQLines = ', qTexts)
#    print()
    i = 0
    for line in q:
        line = line.strip()
        strtemp += line
        strtemp += " "
        qReview.append(line)
        i += 1
    qTexts.append(strtemp)
    strtemp = ""
    return(qReview, qDocnum, qTexts)

# Generate query corpus
# def getQCorp(qTexts)
# return(qCorp)


def getQCorp(qTexts):
    strtemp = ""
    for i in range(len(qTexts)):
        strtemp += qTexts[i]
        qCorp.append(strtemp)
        strtemp = ""
    return(qCorp)

# clean query corpus

# def getQClean(qCorp):
# return(qClean, qLen, fdistQ, fdistQLen,
#            freq_word_Q, freq_word_Qorpus)

# for ea q: tokenize, clean, stem, lem, stopwords, \
# shortwords, etc.


def getQClean(qCorp):
    tokens = str(qCorp)
    # lowecases for content analytics ... we assume, for example, \
    # LOVE is sames love
    tokens = tokens.lower()
    # the dataset contains useless characters and numbers
    # Remove useless numbers and alphanumerical words
    # use regular expression ... a-zA-Z0-9 refers to all English \
    # characters (lowercase & uppercase) and numbers
    # ^a-zA-Z0-9 is opposite of a-zA-Z0-9
    tokens = re.sub("[^a-zA-Z0-9]", " ", tokens)
    # tokenization or word split
    tokens = word_tokenize(tokens)
    # Filter non-alphanumeric characters from tokens
    tokens = [word for word in tokens if word.isalpha()]
    # remove short words
    tokens = [word for word in tokens if len(word) > 2]
    # remove common words
    stoplist = stopwords.words('english')
    # if you want to remove additional words EXAMPLE
    #     more = set(['much', 'even', 'time', 'story'])
    # more = set(['the'])
    # stoplist = set(stoplist) | more
    stoplist = set(stoplist) | stopwords_from_file
    stoplist = set(stoplist)
    tokens = [word for word in tokens if word not in stoplist]
    # stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # -----CLEANING COMPLETE-----

    qClean.append(tokens)
    lenQTokens = len(tokens)
    qLen.append(lenQTokens)
    qfdist = nltk.FreqDist(tokens)

    fdistQ.append(qfdist)
    lenQFdist = len(qfdist)
    fdistQLen.append(lenQFdist)

    # prepare the results of word frequency on corpus data as a list
    freq_word_Q = []

    # two values or columns in fdist_a
    j = 0
    for k, v in qfdist.items():
        freq_word_Q.append([k, v])
        j += 1

    # make it like an Excel worksheet
    # wordlist = pd.DataFrame(freq_word)
    qwordlist = pd.DataFrame(freq_word_Q)

    # pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_rows', 10)

    qwordlistSorted = qwordlist.sort_values(by=[1, 0],
                                            ascending=[False, True])

#    freq_word_Q.append(qwordlistSorted)
    freq_word_Qorpus.append(qwordlistSorted)
    return(qClean, qLen, fdistQ, fdistQLen,
           freq_word_Q, freq_word_Qorpus)

# def getQTuples(freq_word_Q):
# return(q_tuple_words, q_tuple_freq_i)


def getQTuples(freq_word_Q):
    q_tuple_words = tuple([val for (key, val) in enumerate([val for elem in
                           freq_word_Q for val in elem]) if key % 2 == 0])
    q_tuple_freq_i = tuple([val for (key, val) in enumerate([val for elem in
                            freq_word_Q for val in elem]) if key % 2 != 0])
    return(q_tuple_words, q_tuple_freq_i)

# def intersection(post_word_keys):
#     return(docid_set)


def intersection(post_word_keys):
    sets = []
    sets = post_word_keys
#    print('\nsets = ', sets)
    docid_set = (ft.reduce(set.union, [s for s in sets]))
#    print('\ndocid_set = ', docid_set)
    return(docid_set)

# def intersect(a, b):
#     return(c)


def intersect(a, b):
    if len(a) > len(b):
        a, b = b, a
    c = set()
    for x in a:
        if x in b:
            c.add(x)
    return(c)

# generate list of relevant documents

# def getRetDoc(postings, q_tuple_words)
# return(retDoc)


def getRetDoc(postings, q_tuple_words):
    post_word_keys = ([set(postings[word].keys()) for word in q_tuple_words])
    docid_set = intersection(post_word_keys)
    retDoc = docid_set
    return(retDoc)

# def getCosSim(docid, q_tuple_words,
#               q_tuple_freq_i, fdistCorpus):
# return(cosSim)


def getCosSim(docid, q_tuple_words,
              q_tuple_freq_i, fdistCorpus):
    similarity = 0.0
    cosSim = 0.0
    qTF = 0
    qIDF = 0.0
    qWeight = 0.0
    qWeightSquared = 0.0
    qSumWeightSquared = 0.0
    global qVecLen
    docWordWeight = 0.0
    x = 0
    for word in q_tuple_words:
        if word in fdistCorpus[0]:
            qTF = q_tuple_freq_i[x]
            qIDF = getIDF(word)
            qWeight = qTF * qIDF
            qWeightSquared = qWeight**2
            qSumWeightSquared += qWeightSquared
            docWordWeight = getWeight(word, docid)
            similarity += qWeight * docWordWeight
        x += 1
    qVecLen = math.sqrt(qSumWeightSquared)
    cosSim = similarity / (qVecLen * docVecLen[docid])
    return(cosSim)

# def getCosSimScoresList(retDoc, q_tuple_words,
#                         q_tuple_freq_i, fdistCorpus):
# return(cosSimScoresList)


def getCosSimScoresList(retDoc, q_tuple_words,
                        q_tuple_freq_i, fdistCorpus):
    cosSimScoresList = [
        (docid+1, getCosSim(docid, q_tuple_words, q_tuple_freq_i, fdistCorpus))
        for docid in retDoc]
    return(cosSimScoresList)


# ##########################
# Step 4: Ranking Functions
# ##########################

# def getRankCosSimList(cosSimScoresList)
# return(rankCosSimList)


def getRankCosSimList(cosSimScoresList):
    rankCosSimList = sorted(cosSimScoresList, key=lambda l: l[1], reverse=True)
#    print('\nrankCosSimList = \n', rankCosSimList)
    return(rankCosSimList)

#    def getRankListPerQ(qNum, queries_from_file,
#                        postings, fdistCorpus)
#    return(rankListPerQ)


def getRankListPerQ(qNum, queries_from_file,
                    postings, fdistCorpus):

    global q
    q = []
    global qReview
    qReview = []
    global qDocnum
    qDocnum = []
    global qTexts
    qTexts = []
    global qCorp
    qCorp = []
    global qClean
    qClean = []
    global qLen
    qLen = []
    global fdistQ
    fdistQ = []
    global fdistQLen
    fdistQLen = []
    global freq_word_Q
    freq_word_Q = []
    global freq_word_Qorpus
    freq_word_Qorpus = []
    global retDoc
    retDoc = []
    cosSimScoresList = defaultdict(float)
    global rankCosSimList
    rankCosSimList = []

    query = []
    input = ""
    input = queries_from_file[qNum]
    query.append(input)
    q = query

    # def getQLines(q)
    # return(qReview, qDocnum, qTexts)
    qReview, qDocnum, qTexts = getQLines(q)

    # def getQCorp(qTexts)
    # return(qCorp)
    qCorp = getQCorp(qTexts)

    # for ea qCorp: tokenize, clean, stem, lem, stopwords,\
    # shortwords, etc.

    # def getQClean(qCorp):
    # return(qClean, qLen, fdistQ, fdistQLen,
    #            freq_word_Q, freq_word_Qorpus)
    qClean, qLen, fdistQ, fdistQLen, freq_word_Q, freq_word_Qorpus\
        = getQClean(qCorp)

    # def getQTuples(freq_word_Q):
    # return(q_tuple_words, q_tuple_freq_i)
    q_tuple_words, q_tuple_freq_i = getQTuples(freq_word_Q)

    # def getRetDoc(postings, q_tuple)
    # return(retDoc)
    retDoc = getRetDoc(postings, q_tuple_words)

    cosSimScoresList = getCosSimScoresList(retDoc, q_tuple_words,
                                           q_tuple_freq_i, fdistCorpus)

    # def getRankCosSimList(cosSimScoresList)
    # return(rankCosSimList)
    rankCosSimList = getRankCosSimList(cosSimScoresList)

    rankListPerQ = rankCosSimList

    return(rankListPerQ)

# def sendToOutputFolder(dir_path_output, output_qid_docid)
# return()

# sendToOutputFolder(dir_path_output, output_qid_docid)


def sendToOutputFolder(dir_path_output, output_qid_docid):
    files_ouput = os.listdir(dir_path_output)
    for fo in files_ouput:
        with open(dir_path_output+'/'+os.path.basename(fo), 'w') as ofile:
            ofile.write('\n'.join('{} {}'.format(qiddocid[0], qiddocid[1]) for
                                  qiddocid in output_qid_docid))
    return()

# def sendToOutputRecPrec(dir_path_RecPrec, recPrec)
# return()

# sendToOutputRecPrec(dir_path_RecPrec, recPrec)


def sendToOutputRecPrec(dir_path_RecPrec, recPrec):
    files_RecPrec = os.listdir(dir_path_RecPrec)
    for frp in files_RecPrec:
        with open(dir_path_RecPrec+'/'+os.path.basename(frp), 'w') as frpfile:
            frpfile.write('\n'.join('{} {} {} {} {} {} {} {} {} {}'.
                          format(rpVal.TopXDocs, rpVal.QNum,
                          rpVal.TxQ, rpVal.RelIddxRankNum, rpVal.OQD_QNum,
                          rpVal.OQD_DocID, rpVal.Relevant, rpVal.RelCnt,
                          rpVal.Recall, rpVal.Precision)
                          for rpVal in recPrec))
    return()

# def getRelevance(dir_path_relevance):
# return(relevance_from_file)

# relevance_from_file = getRelevance(dir_path_relevance)


def getRelevance(dir_path_relevance):
    files_relevance = os.listdir(dir_path_relevance)
    for fr in files_relevance:
        with open(dir_path_relevance+'/'+os.path.basename(fr), 'r') as rfile:
            relevance_from_file = [tuple(int(n) for n in line.split())
                                   for line in rfile]
    return(relevance_from_file)

# def getQtyRelDocPerQ(relevance_from_file):
# return(qtyRelDocPerQ)

# qtyRelDocPerQ = getQtyRelDocPerQ(relevance_from_file)


def getQtyRelDocPerQ(relevance_from_file):
    qNum = 1
    qtyRelDoc = 0
    totRelDoc = len(relevance_from_file)
    for n in range(totRelDoc):  # xrange
        if (n == totRelDoc - 1) & (relevance_from_file[n][0] == qNum):
            qtyRelDoc += 1
            qtyRelDocPerQ.append(qtyRelDoc)
        elif relevance_from_file[n][0] == qNum:
            qtyRelDoc += 1
        elif relevance_from_file[n][0] == qNum + 1:
            qtyRelDocPerQ.append(qtyRelDoc)
            qtyRelDoc = 1
            qNum += 1
    return(qtyRelDocPerQ)

# def getAvg(queryNumber, pointValue):
# return(average)

# p.append(getAvg(intPrecStart, pVal))


def getAvg(queryNumber, pointValue):
    i = queryNumber
    total = 0
    for i in range(i, 10 + i):  # xrange
        total = total + intPrec[i][pointValue]
    average = total / 10
    return(average)

# #############################################################################
# MAIN
# #############################################################################


if __name__ == '__main__':
    print()

    # *************************
    # ---BEGIN Declare Variables---
    # *************************

    # -----------------
    # constants
    # ______path to cranfieldDocs directory_____
    # dir_path = 'C:/Users/Derek Christensen/Dropbox/_cis833irtm/hw2//\
    # cranfieldDocs'
    # dir_path = 'C:/Users/Derek Christensen/Dropbox/_cis833irtm/hw2/data'
    # dir_path = 'C:/Users/Derek Christensen/Dropbox/_cis833irtm/hw2/data-temp'
    # dir_path = 'C:/Users/Derek Christensen/Dropbox/_cis833irtm/hw2/data-misc'

#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/cranfieldDocs'

#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-1'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-fox'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-fox2'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-15'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-Q2'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-Q2-2'
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/data-Q2-3'

    dir_path_stopwords = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/stopwords'

    dir_path_queries = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/queries'
#    dir_path_queries = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/queries2'
#    dir_path_queries = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/Q2'
#    dir_path_queries = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/Q6'

    dir_path_output = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/output'

    dir_path_RecPrec = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/RecPrec'

    dir_path_relevance = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/relevance'
#    dir_path_relevance = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/rel-Q2'
#    dir_path_relevance = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/rel-Q2-2'
#    dir_path_relevance = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/rel-Q123'
#    dir_path_relevance = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/rel-Q6'

    # declare arrays, variables

    global stopwords_from_file
    global queries_from_file

    # def getFiles(dir_path)
    # return(files, file_names, file_idx, file_zip, file_dict, file_dict_enum)
    files = []
    file_names = []
    file_idx = []
    file_zip = []
    file_dict = []
    file_dict = {}
    file_dict_enum = {}

    # def getLines(files, dir_path)
    # return(review, docnum, titles, texts)
    review = []
    docnum = []
    titles = []
    texts = []
    tf = []
    j = 0

    # def getPerDocCorp(titles, texts)
    # return(perDocCorp, corpus)
    perDocCorp = []
    corpus = []

    # def getPerDocCorpClean(perDocCorp)
    # return(perDocCorpClean, perDocLen, fdistPerDoc, fdistPerDocLen,
    #       freq_word_PerDoc)
    perDocCorpClean = []
    perDocLen = []
    fdistPerDoc = []
    fdistPerDocLen = []
    freq_word_PerDoc = []

    # def getCorpusClean(corpus)
#     return(corpusClean, corpusLen, fdistCorpus, fdistCorpusLen,
#            freq_word_Corpus)
    corpusClean = []
    corpusLen = []
    fdistCorpus = []
    fdistCorpusLen = []
    freq_word_Corpus = []

    # def getPostings(file_names, freq_word_PerDoc, perDocCorpClean)
    # return(postings)
    postings = defaultdict(dict)

    # getDF(file_names, freq_word_Corpus, postings)
    # return(df)
    df = defaultdict(int)

    # getDocVecLen(file_names, freq_word_Corpus)
    # return(docVecLen)
    docVecLen = defaultdict(float)

    # def getRankListPerQ(qNum, queries_from_file,
    #                     postings, fdistCorpus)
    # return(rankListPerQ)
    rankListPerQ = []
    global output_qid_docid
    output_qid_docid = []

    # def getQtyRelDocPerQ(relevance_from_file):
    # return(qtyRelDocPerQ)

    global qtyRelDocPerQ
    qtyRelDocPerQ = []

    # def getAvg(queryNumber, pointValue):
    # return(average)

    avgPrec10 = []
    avgPrec50 = []
    avgPrec100 = []
    avgPrec500 = []

    # *************************
    # ---END Declare Variables---
    # *************************

# #############################################################################
# Step 0. Input Data Paths From User
# #############################################################################

    # Makes a function that will contain the
    # desired program.
    def getDataPath(folderName, inputPath, path):
        print('folderName = ', folderName)
        print('inputPath = ', inputPath)
#        inputPath = raw_input('Enter path to the ', folderName, 'folder:')

#        inputPath = (raw_input('Enter path to the ', folderName, 'folder:'))
#        inputPath = sys.stdin.readline('Enter path to the ', folderName, 'folder:')

#        inputPath = raw_input("Enter path to the ", folderName, "folder:")

        print('Enter path to the ", folderName, "folder.')
        print('In the exact form of : ')
        print()
        print('r\'C:/Users/derekc/Dropbox/__cis833irtm/hw2/cranfieldDocs\'')
        print()
        print("Including the r and single apostrophe at the beginnning and the single apostrophe at the end.")

        inputPath = raw_input("Enter: ")
        print()
        print(inputPath)

#        path.append(inputPath)
#        return(path)
        return(inputPath)


# ----- A COMPLETE EXAMPLE ---- #################################
## Makes a function that will contain the
## desired program.
#def example():
#
#    # Calls for an infinite loop that keeps executing
#    # until an exception occurs
#    while True:
#        test4word = input("What's your name? ")
#
#        try:
#            test4num = int(input("From 1 to 7, how many hours do you play in your mobile?" ))
#
#        # If something else that is not the string
#        # version of a number is introduced, the
#        # ValueError exception will be called.
#        except ValueError:
#            # The cycle will go on until validation
#            print("Error! This is not a number. Try again.")
#
#        # When successfully converted to an integer,
#        # the loop will end.
#        else:
#            print("Impressive, ", test4word, "! You spent", test4num*60, "minutes or", test4num*60*60, "seconds in your mobile!")
#            break
#
## The function is called
#example()

    path = []
    folderName = 'cranfieldDocs'
    inputPath = 'false'
    print('folderName = ', folderName)
    print('inputPath = ', inputPath)
#    dir_path = r'C:/Users/derekc/Dropbox/__cis833irtm/hw2/cranfieldDocs'
    dir_path = getDataPath(folderName, inputPath, path)

    print()
    print('dir_path = ', dir_path)
    print()



# #############################################################################
# Step 1. Preprocessing
# #############################################################################

    # ####################################
    # ###### OBTAINING DATA FILES ########
    # ####################################

    # *************************
    # ---BEGIN getStopwords---
    # *************************

    # def getStopwords(dir_path_stopwords)
    # return(stopwords_from_file)

    stopwords_from_file = getStopwords(dir_path_stopwords)

    # *************************
    # ---END getStopwords---
    # *************************

    # ************************
    # ----BEGIN OF getFiles---
    # ************************
    # get all files inside the directory & process to arrays & dicts
    # getFiles(dir_path, files, file_names, file_idx, file_zip, file_dict)
    # print(files)
    #
    # def getFiles(dir_path)
    # return(files, file_names, file_idx, file_zip, file_dict, files_dict_enum)

    files, file_names, file_idx, file_zip, file_dict, file_dict_enum = \
        getFiles(dir_path)

    # *************************
    # ----END OF getFiles-----
    # *************************

    # ###########################################################
    # ###### Eliminate SGML tags & only keep TITLE & TEXT #######
    # ###########################################################

    # ###########################################################
    # ####### READING DATA AS LIST & ELIMINATE SGML TAGS ########
    # ###########################################################

    # *************************
    # -----BEGIN getLines------
    # *************************
    # start processing the ipfile & break all files into lines
    #
    # def getLines(files, dir_path):
    # return(review, docnum, titles, texts)

    review, docnum, titles, texts = getLines(files, dir_path)

    # DONE ASSIGNING DOCNUM TITLES AND TEXTS

    # *************************
    # -------END getLines------
    # *************************

    # ###########################################################
    # ########### Merge Titles and Texts ############
    # ###########################################################

    # #####################################
    # ###### GET EACH DOCs CORPUS #########
    # #####################################

    # *************************
    # ---BEGIN getPerDocCorp---
    # *************************
    # merge TITLES & TEXTS into 1 STRING per DOC

    # def getPerDocCorp(titles, texts)
    # return(perDocCorp, corpus)

    perDocCorp, corpus = getPerDocCorp(titles, texts)

    # *************************
    # ----END getPerDocCorp----
    # *************************

    # ###########################################################
    # ########### Clean Corpus ############
    # ###########################################################

    # #######################################
    # ###### CLEAN EACH DOCs CORPUS #########
    # #######################################

    # *************************
    # --BEGIN perDocCorpClean--
    # *************************
    # for ea perDocCorp: tokenize, clean, stem, lem, stopwords, \
    # shortwords, etc.

    # def getPerDocCorpClean(perDocCorp)
    # return(perDocCorpClean, perDocLen, fdistPerDoc, fdistPerDocLen,
    #   freq_word_PerDoc)

    perDocCorpClean, perDocLen, fdistPerDoc, fdistPerDocLen, freq_word_PerDoc\
        = getPerDocCorpClean(perDocCorp)

    # *************************
    # --END perDocCorpClean--
    # *************************

    # #######################################
    # ######## CLEAN ENTIRE CORPUS ##########
    # #######################################

    # *************************
    # --BEGIN corpusClean--
    # *************************
    # for corpus: tokenize, clean, stem, lem, stopwords, \
    # shortwords, etc.

    # def getCorpusClean(corpus)
    # return(corpusClean, corpusLen, fdistCorpus, fdistCorpusLen,
    #        freq_word_Corpus)

    corpusClean, corpusLen, fdistCorpus, fdistCorpusLen, freq_word_Corpus\
        = getCorpusClean(corpus)

    # *************************
    # --END corpusClean--
    # *************************

# #############################################################################
# Step 2: Indexing
# #############################################################################

    # *************************
    # --BEGIN Create Postings Dictionary--
    # *************************

    # create Postings dictionary

    # def getPostings(file_names, freq_word_PerDoc, perDocCorpClean)
    # return(postings)

#    postings = defaultdict(dict)

    postings = getPostings(file_names, freq_word_PerDoc, perDocCorpClean)

    # *************************
    # --END Postings Dictionary--
    # *************************

    # #############################################################
    # ########### Create DF Dictionary ############
    # #############################################################

    # *************************
    # --BEGIN Create DF Dictionary--
    # *************************

    # Create DF Dictionary

    # getDF(file_names, freq_word_Corpus, postings)
    # return(df)

#    df = defaultdict(int)

    df = getDF(file_names, freq_word_Corpus, postings)

#    DFfirst20 = {k: df[k] for k in df.keys()[:20]}

    # *************************
    # --END DF Dictionary--
    # *************************

#    print('##########################################################')
#    print('##########################################################')
#    print('#####-----CORPUS & QUERY OUTPUT DIVIDER-----##############')
#    print('##########################################################')
#    print('##########################################################')

    # #############################################################
    # ########### Create Inverted Index & docVecLen ############
    # #############################################################

    # *************************
    # --CREATE Inverted Index & docVecLen--
    # *************************

    # create Inverted Index & docVecLen

    # getDocVecLen(file_names, freq_word_Corpus)
    # return(docVecLen)

#    docVecLen = defaultdict(float)

    docVecLen = getDocVecLen(file_names, freq_word_Corpus)
    docVecLen

    # *************************
    # --END Inverted Index & docVecLen--
    # *************************

# #############################################################################
# Step 3: Retrival
# #############################################################################

    # *************************
    # --BEGIN Get Query from User--
    # *************************

    # *************************
    # --"queries.txt" Input--
    # *************************

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # *************************
    # ---BEGIN getQueries---
    # *************************

    # def getQueries(dir_path_queries)
    # return(queries_from_file)

    queries_from_file = getQueries(dir_path_queries)

    # *************************
    # ---END getQueries---
    # *************************

    # ############################################################
    # ########### Reading Query as List ############
    # ############################################################

    # ############################################################
    # ########### Start Processing Q and Break Into Lines ############
    # ############################################################

    # *************************
    # ---BEGIN getQLines---
    # *************************

#    # start processing q and break into lines
#
#    # def getQLines(q)
#    # return(qReview, qDocnum, qTexts)
#
#    qReview, qDocnum, qTexts = getQLines(q)

    # *************************
    # ---END getQLines---
    # *************************

    # ##########################################################
    # ########### Get Q Corpus ############
    # ##########################################################

    # #####################################
    # ###### Merge TITLES & TEXTS into 1 String per Query #########
    # #####################################

    # *************************
    # ---BEGIN getQCorp---
    # *************************

#    # merge TITLES & TEXTS into 1 STRING per Query
#
#    # def getQCorp(qTexts)
#    # return(qCorp)
#
#    qCorp = getQCorp(qTexts)

    # *************************
    # --END getQCorp--
    # *************************

    # ############################################################
    # ########### Clean Q Corpus ############
    # ############################################################

    # #######################################
    # ###### For ea qCorp: tokenize, clean, stem, lem, stopwords,  #########
    # ###### shortwords, etc.  #########
    # #######################################

    # *************************
    # --BEGIN getQClean--
    # *************************

#    # for ea qCorp: tokenize, clean, stem, lem, stopwords,\
#    # shortwords, etc.
#
#    # def getQClean(qCorp):
#    # return(qClean, qLen, fdistQ, fdistQLen,
#    #            freq_word_Q, freq_word_Qorpus)
#
#    qClean, qLen, fdistQ, fdistQLen, freq_word_Q, freq_word_Qorpus\
#        = getQClean(qCorp)

    # *************************
    # --END getQClean--
    # *************************

    # ############################################################
    # ### Generate Tuples for Both the Query's Words & Freqs #####
    # ############################################################

    # *************************
    # --BEGIN Query Word & Freq Tuples--
    # *************************

#    # generate tuples for both the Query's Words & Freqs
#    #
#    # def getQTuples(freq_word_Q):
#    # return(q_tuple_words, q_tuple_freq_i)
#
#    q_tuple_words, q_tuple_freq_i = getQTuples(freq_word_Q)

    # *************************
    # --END Query Word & Freq Tuples--
    # *************************

    # ############################################################
    # ######## Generate List of Relevant Documents #########
    # ############################################################

    # *************************
    # --BEGIN Document Retrieval--
    # *************************

#    # generate list of relevant documents
#
#    # def getRetDoc(postings, q_tuple)
#    # return(retDoc)

#    retDoc = getRetDoc(postings, q_tuple_words)

    # *************************
    # --END Document Retrieval--
    # *************************

    # ##########################################################
    # ######## Calculate CosSim Scores b/t Q & Ea. Doc ########
    # ##########################################################

    # *************************
    # --BEGIN getCosSimScoresList--
    # *************************

#    # calculate CosSim Scores b/t q & ea. doc

#    # def getCosSimScoresList(retDoc, q_tuple_words,
#    #                         q_tuple_freq_i, fdistCorpus):
#    # return(cosSimScoresList)

#    #     def getCosSim(docid, q_tuple_words, q_tuple_freq_i):
#    #     return(cosSim)

#    # cosSimScoresList = defaultdict(float)

#    cosSimScoresList = getCosSimScoresList(retDoc, q_tuple_words,
#                                           q_tuple_freq_i, fdistCorpus)

#    # cosSimScoresList values

    # *************************
    # --END getCosSimScoresList--
    # *************************

# #############################################################################
# Step 4: Ranking
# #############################################################################

    # *************************
    # --BEGIN getRankCosSimList--
    # *************************

#    # rank list of relevant documents

#    # def getRankCosSimList(cosSimScoresList)
#    # return(rankCosSimList)

#    # rankCosSimList = []

#    rankCosSimList = getRankCosSimList(cosSimScoresList)

    # *************************
    # --END getRankCosSimList--
    # *************************

    # *************************
    # --BEGIN getRankListPerQ--
    # *************************

    # rank list of relevant documents per query

#    def getRankListPerQ(qNum, queries_from_file,
#                        postings, fdistCorpus)
#    return(rankListPerQ)

#    rankListPerQ = []
#    output_qid_docid = []

    for qNum in range(len(queries_from_file)):
        rankListPerQ = getRankListPerQ(qNum, queries_from_file,
                                       postings, fdistCorpus)

        for relvDocIdx in range(len(rankListPerQ)):
            output_qid_docid.append((qNum + 1, rankListPerQ[relvDocIdx][0]))

    # *************************
    # --END getRankListPerQ--
    # *************************

    # *************************
    # ---BEGIN sendToOutputFolder---
    # *************************

    # def sendToOutputFolder(dir_path_output, output_qid_docid)
    # return()

    sendToOutputFolder(dir_path_output, output_qid_docid)

    # *************************
    # ---END sendToOutputFolder---
    # *************************

    # *************************
    # ---BEGIN getRelevance---
    # *************************

    # def getRelevance(dir_path_relevance):
    # return(relevance_from_file)

    relevance_from_file = getRelevance(dir_path_relevance)

    # *************************
    # ---END getRelevance---
    # *************************

# def getQTuples(freq_word_Q):
# return(q_tuple_words, q_tuple_freq_i)


# def getQTuples(freq_word_Q):
#    q_tuple_words = tuple([val for (key, val) in enumerate([val for elem in
#                           freq_word_Q for val in elem]) if key % 2 == 0])
#    q_tuple_freq_i = tuple([val for (key, val) in enumerate([val for elem in
#                            freq_word_Q for val in elem]) if key % 2 != 0])
#    return(q_tuple_words, q_tuple_freq_i)

#    global freq_word_Q
#    freq_word_Q = []

    # *************************
    # ---BEGIN getQtyRelDocPerQ---
    # *************************

    # def getQtyRelDocPerQ(relevance_from_file):
    # return(qtyRelDocPerQ)

    qtyRelDocPerQ = getQtyRelDocPerQ(relevance_from_file)

    # *************************
    # ---END getQtyRelDocPerQ---
    # *************************

    # *************************
    # ---BEGIN recPrec---
    # *************************

    # def getQtyRelDocPerQ(relevance_from_file):
    # return(qtyRelDocPerQ)

#    qtyRelDocPerQ = getQtyRelDocPerQ(relevance_from_file)

    STD_RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#    print('\nSTD_RECALL_LEVELS = ', STD_RECALL_LEVELS)

    TOP_X_DOCS = [10, 50, 100, 500]
#    TOP_X_DOCS = [10]
#    TOP_X_DOCS = [20]

    global recPrec
    recPrec = []
    rows = len(output_qid_docid)
    cnt = 0
    for i in range(len(output_qid_docid)):
        if output_qid_docid[i][0] == 1:  # was 2
            cnt += 1

    topDocIdx = 0  # TOP_X_DOCS = [10, 50, 100, 500]
    relIdxRankN = 0  # rows = len(output_qid_docid) == 9402

    for topDocIdx in range(len(TOP_X_DOCS)):  # {0, 1, 2, 3}
        relIdxRankN = 0  # rows = len(output_qid_docid) == 9402
        qNum = 0  # [0:wave, 1:shock, 2:blunt, . . ., 9:lift-drag]
        recall = 0
        precision = 0
        for qNum in range(len(queries_from_file)):  # {0, 1, 2, . . ., 9}
            n = 0  # [10: 0..9, 50: 0..49, 100: 0..99, 500: 0..499]
            recall = 0
            relCnt = 0

            for n in range(TOP_X_DOCS[topDocIdx]):  # {10, 50, 100, 500}
                while output_qid_docid[relIdxRankN][0] != qNum + 1:
                    relIdxRankN += 1
                if output_qid_docid[relIdxRankN][0] == qNum + 1:
                    if output_qid_docid[relIdxRankN] in relevance_from_file:
                        rel = 1
                        relCnt += 1
                        recall = relCnt / qtyRelDocPerQ[qNum]
                        precision = relCnt / (n + 1)
                    else:
                        rel = 0
                        precision = relCnt / (n + 1)

                    recPrec.append(Docrow(TOP_X_DOCS[topDocIdx], qNum + 1,
                                          n + 1, relIdxRankN,
                                          output_qid_docid[relIdxRankN][0],
                                          output_qid_docid[relIdxRankN][1],
                                          rel, relCnt, recall, precision))
                relIdxRankN += 1

    # *************************
    # ---BEGIN sendToOutputRecPrec---
    # *************************

    # def sendToOutputRecPrec(dir_path_RecPrec, output_qid_docid)
    # return()

    sendToOutputRecPrec(dir_path_RecPrec, recPrec)

#    def sendToOutputRecPrec(dir_path_RecPrec, recPrec):
#        files_RecPrec = os.listdir(dir_path_RecPrec)
#        for frp in files_RecPrec:
#            with open(dir_path_RecPrec+'/'+os.path.basename(frp), 'w') as frpfile:
#                frpfile.write('\n'.join('{} {} {} {} {} {} {} {} {} {}'.
#                              format(rpVal.TopXDocs, rpVal.QNum,
#                              rpVal.TxQ, rpVal.RelIddxRankNum, rpVal.OQD_QNum,
#                              rpVal.OQD_DocID, rpVal.Relevant, rpVal.RelCnt,
#                              rpVal.Recall, rpVal.Precision)
#                              for rpVal in recPrec))
#        return()

    # *************************
    # ---END sendToOutputRecPrec---
    # *************************


    # *************************
    # ---END recPrec---
    # *************************

    # *************************
    # ---BEGIN intPrec---
    # *************************

    intPrec = []
    intPrecIdx = 0
    recPrecIdx = 0  # {0, 1, 2, . . ., 6599}
    topDocIdx = 0  # {0, 1, 2, 3}

    for topDocIdx in range(len(TOP_X_DOCS)):  # {4}
        numRowsPerQ = TOP_X_DOCS[topDocIdx]  # {10, 50, 100, 500}
        qNum = 0  # [0:wave, 1:shock, 2:blunt, . . ., 9:lift-drag]

        for qNum in range(len(queries_from_file)):  # {0, 1, 2, . . ., 9}
            firstRecPrecIdx = recPrecIdx
            lastRecPrecIdx = recPrecIdx + (numRowsPerQ - 1)
            pointer = firstRecPrecIdx
            #  current high RECALL for qNum
            sRL = 0
            intPrec.append([])

            for sRL in range(len(STD_RECALL_LEVELS)):  # [11]
                while (STD_RECALL_LEVELS[sRL] > recPrec[pointer][8]):
                    if pointer < lastRecPrecIdx:
                        pointer += 1
                    else:
                        break
                if (sRL == 0) or (STD_RECALL_LEVELS[sRL] <
                                  recPrec[pointer][8]):
                    maxPrec = pointer
                    #  current high PRECISION for qNum
                    j = 0
                    for j in range(maxPrec, lastRecPrecIdx):
                        if recPrec[maxPrec][9] <= recPrec[j + 1][9]:
                            maxPrec = j + 1
                    pointer = maxPrec
                    intPrec[intPrecIdx].append(recPrec[maxPrec].Precision)
                elif (sRL != 0) and (STD_RECALL_LEVELS[sRL] ==
                                     recPrec[pointer][8]):
                    maxPrec = pointer
                    #  current high PRECISION for qNum
                    j = 0
                    for j in range(maxPrec, lastRecPrecIdx):
                        if recPrec[maxPrec][9] <= recPrec[j + 1][9]:
                            maxPrec = j + 1
                    pointer = maxPrec
                    intPrec[intPrecIdx].append(recPrec[maxPrec][9])
                elif (sRL != 0) and (STD_RECALL_LEVELS[sRL] >
                                     recPrec[pointer][8]):
                    intPrec[intPrecIdx].append(0)
            intPrecIdx += 1
            recPrecIdx = pointer + 1

    print()
    for j in range(len(intPrec)):
        print('intPrec[', j, '] = ', intPrec[j])
    print()
    print()

    # *************************
    # ---END intPrec---
    # *************************

    # *************************
    # ---BEGIN avgPrecPerQuery---
    # *************************

    print('avgPrec10 = ', avgPrec10)
    print('avgPrec50 = ', avgPrec50)
    print('avgPrec100 = ', avgPrec100)
    print('avgPrec500 = ', avgPrec500)
    print()

    #   STD_RECALL_LEVELS = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #   print('\nSTD_RECALL_LEVELS = ', STD_RECALL_LEVELS)
    #
    #   TOP_X_DOCS = [10, 50, 100, 500]

    for topXdoc in range(len(TOP_X_DOCS)):
        p = []
        intPrecStart = 10 * topXdoc
        pVal = 0
        for pVal in range(len(STD_RECALL_LEVELS)):
            p.append(getAvg(intPrecStart, pVal))
        if topXdoc == 0:
            avgPrec10 = p
        elif topXdoc == 1:
            avgPrec50 = p
        elif topXdoc == 2:
            avgPrec100 = p
        elif topXdoc == 3:
            avgPrec500 = p

    print('avgPrec10 = ', avgPrec10)
    print('avgPrec50 = ', avgPrec50)
    print('avgPrec100 = ', avgPrec100)
    print('avgPrec500 = ', avgPrec500)

    # *************************
    # ---END avgPrecPerQuery---
    # *************************

    # *************************
    # ---BEGIN printPlot---
    # *************************

    #   STD_RECALL_LEVELS = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

#    avgPrec10	=	0.575833	0.330000	0.317143	0.146667	0.146667	0.129167	0.000000	0.000000	0.000000	0.000000	0.000000
#    avgPrec50	=	0.600476	0.401331	0.354286	0.238505	0.204433	0.160537	0.102345	0.066275	0.033333	0.000000	0.000000
#    avgPrec100	=	0.600476	0.401331	0.361429	0.245648	0.227825	0.198445	0.129098	0.095312	0.071406	0.009474	0.009474
#    avgPrec500	=	0.604643	0.405498	0.361659	0.245878	0.236095	0.206714	0.143885	0.108551	0.084644	0.031209	0.022555

#    # x axis values
#    x = STD_RECALL_LEVELS
#
#    # corresponding y axis values
#    y = [0.5758333333333333, 0.32999999999999996, 0.3171428571428571, 0.14666666666666667, 0.14666666666666667, 0.12916666666666665, 0.0, 0.0, 0.0, 0.0, 0.0]
#
#    # plotting the points
#    plt.plot(x, y)

    # line 1 points
    x1 = STD_RECALL_LEVELS
    y1 = avgPrec10
    # plotting the line 1 points
    plt.plot(x1, y1, label="Top 10 Docs", marker='o', markersize=4)

    # line 2 points
    x2 = STD_RECALL_LEVELS
    y2 = avgPrec50
    # plotting the line 2 points
    plt.plot(x2, y2, label="Top 50 Docs",  marker='s', markersize=4)

    # line 3 points
    x3 = STD_RECALL_LEVELS
    y3 = avgPrec100
    # plotting the line 3 points
    plt.plot(x3, y3, label="Top100 Docs", marker='^', markersize=4)

    # line 4 points
    x4 = STD_RECALL_LEVELS
    y4 = avgPrec500
    # plotting the line 4 points
    plt.plot(x4, y4, label="Top 500 Docs",  marker='*', markersize=10)

    # setting x and y axis range
    plt.ylim(0, 0.7)
    plt.xlim(0, 1)

    # naming the x axis
    plt.xlabel('RECALL')
    # naming the y axis
    plt.ylabel('PRECISION')

    # giving a title to my graph
    plt.title('AVERAGE PRECISION-RECALL CURVES')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

    # *************************
    # ---END printPlot---
    # *************************
