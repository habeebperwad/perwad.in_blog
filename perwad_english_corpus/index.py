#!/usr/bin/python3

import glob, re
import os.path
import nltk
import numpy as np
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer


LEMMA_FILE_EXT  = "lemma"
TEMP_DIR = "tmp/"
LEMMA_COUNT_FILE = TEMP_DIR + "lemma_count.txt"
LEMMAX_COUNT_FILE = TEMP_DIR + "lemma_countx.txt"
lemmatizer = WordNetLemmatizer()


def get_lemma(wl):
    tag = wl[1][0].lower()
    pos = tag if tag in ['a', 'r', 'n', 'v'] else None
    return lemmatizer.lemmatize(wl[0], pos) if pos else wl[0]


def tmpfilename(file_with_path, ext):
    return TEMP_DIR + file_with_path.replace("/", "__") + "." + ext


def generate_lemmas():
    print("\nGenarate lemma data for each document...")
    documents = glob.glob("documents/*/*.txt")
    for doc in documents:
        lemma_file = tmpfilename(doc, LEMMA_FILE_EXT)

        # If the lemma is already created, just skip.
        if (os.path.isfile(lemma_file)):
            continue
            pass

        print(doc, end=" "*20 + "\r")
        with open(doc) as f:
           file_content = f.read()
        sentences = nltk.sent_tokenize(file_content)
        lemmas = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(words)
            for k in tags:
                lemmas.append(get_lemma(k).lower())

        # Save the lemma in file
        file = open(lemma_file, 'w')
        file.write(str(lemmas))
        file.close()

    # Move the screen curson to next line.
    print(" " * 60 + "\n")

def count_all_lemmas():

    print("\nCount lemmas from all documensts...")
    documents = glob.glob(TEMP_DIR+"*." + LEMMA_FILE_EXT)
    lemmas = []
    for doc in documents:
        print(doc, end=" "*20 + "\r")
        with open(doc) as f:
            lemma = f.read()
        lemmas.extend(eval(lemma))

    print(" " * 60 + "\n")

    c = Counter(lemmas)
    print("Here is the most common all types of lemmas:")
    print(c.most_common(10))
    file = open(LEMMA_COUNT_FILE, 'w')
    file.write(str(c.most_common()))
    file.close()


def count_all_lemmas_with_only_aphabets():
    
    with open(LEMMA_COUNT_FILE) as f:
        lemma_count = eval(f.read())
    alpha_lemmas = [word for word in lemma_count if re.search('^[a-z]+$', word[0])]
    return alpha_lemmas


def print_statistics():

    ccount = [10,100,1000,2000,3000,4000,5000,12500]
    lemmas = count_all_lemmas_with_only_aphabets()
    print("\nCount of all lemmas: " + str(sum([l[1] for l in lemmas])))
    print("\nNumber of different lemmas: " + str(len(lemmas)))
    total_lemmas = sum([w[1] for w in lemmas])
    print("\nMost common lemma count and share in all lemmas")
    print("COUNT","\t", "%")
    for c in ccount:
        print(c, "\t",round(sum([h[1] for h in lemmas[0:c]])/total_lemmas*100, 1))
   
    # save lemmax for R to plot graph
    lemmax = []
    for c in [w for w in range(10,10001,10)]:#ccount:
        lemmax.append((c,round(sum([h[1] for h in lemmas[0:c]])/total_lemmas*100, 1)))
    np.savetxt(LEMMAX_COUNT_FILE,(lemmax), delimiter=',', fmt="%s")

#
# START READING FROM HERE :)
#
generate_lemmas()
count_all_lemmas()
print_statistics()
