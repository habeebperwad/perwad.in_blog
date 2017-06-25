import glob, math, json, re
import nltk, matplotlib.pyplot as plt


porter = nltk.PorterStemmer()
stop_word_list = set(nltk.corpus.stopwords.words('english'))
with open("common_words.json") as f:
    common_word_list= json.load(f)


# Just return all words.
def all_words(words):
    return words

# Return all words except stop-words.
def no_stop_words(words):
    return [word for word in words if word not in stop_word_list]

# Return only common words. 
def common_words(words):
    return [word for word in words if word in common_word_list]

# Return only stop-words.
def stop_words(words):
    return [word for word in words if word in stop_word_list]

# Return stemm of words.
def stemming(words):
    return set([porter.stem(word) for word in words])


data_cach = {}
def get_data_cached(type):

    if(type in data_cach):  # return if cached data available.
        return data_cach[type]

    print("Reading the data of type: " + type)
    data =[]
    docs = sorted(glob.glob("data/%s/*" % type))
    for doc in docs:
        with open(doc) as f:
            content = f.read();
        textInLowerCase = content.lower()
        textCleaned = re.sub('[^a-z]+', ' ', textInLowerCase)
        words = textCleaned.split()
        data.append(set(words))

    data_cach[type] = data  # catch it.
    return data


def get_data(type, wordProcessor):
    data = get_data_cached(type)
    ret = []
    for d in data:
        ret.append({'category': type , 'words': wordProcessor(d)})
    return ret


def get_profile_data_and_obituary_data(wordProcessor):
    return get_data('profile', wordProcessor), get_data('obituary', wordProcessor)


def document_features(document, word_features):
    document_words = set(document['words'])
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = word in document_words
    return features


def get_training_data_and_test_data(wordProcessor, total_training_count):
    training_count_per_category = int(total_training_count/2)
    profile_data, obituary_data = get_profile_data_and_obituary_data(wordProcessor)
    training_data = profile_data[:training_count_per_category]
    training_data.extend(obituary_data[:training_count_per_category])
    test_data = profile_data[training_count_per_category:] 
    test_data.extend(obituary_data[training_count_per_category:])
    return training_data, test_data


def process(wordProcessor, training_data_count):
    training_data, test_data = get_training_data_and_test_data(
            wordProcessor, training_data_count)
    word_features = set([w for x in training_data for w in x['words']])
    print("Number of features: " + str(len(word_features)))
    training_feature_sets = [(document_features(d, word_features), 
        d['category']) for d in training_data]
    test_feature_sets = [(document_features(d, word_features), d['category']) for d in test_data]
    classifier = nltk.NaiveBayesClassifier.train(training_feature_sets)
    ret = round(100*nltk.classify.accuracy(classifier, test_feature_sets), 2)
    print("Accuracy: %04.2f" % ret)
    classifier.show_most_informative_features(10)
    print("\n")
    return ret


def batch_process(training_data_count):

    ret = {}

    print("# All words used for feature.")
    ret['All words'] = process(all_words, training_data_count)

    print("# No stop words used for feature.")
    ret['No stop words'] = process(no_stop_words, training_data_count)

    print("# Only stop words used for feature.")
    ret['Only stop words'] = process(stop_words, training_data_count)

    print("# Only common words used for feature.")
    ret['Only common words'] = process(common_words, training_data_count)

    print("# Stemmed words used for feature.")
    ret['Stemming'] = process(stemming, training_data_count)

    return ret


def visualize(data, training_data_counts):
    colors = {
        "Only stop words": "r", 
        "Only common words": "b",
        "Stemming": "g", 
        "All words": "c", 
        "No stop words": "y"}
    #plt.title("I couldnt find a title for this plot!")
    plt.ylabel('Accuracy of classification')
    plt.xlabel('Number of examples in training data')

    for k,v in colors.items():
        plt.plot(training_data_counts, [sample[k] for sample in data], v, label=k,)

    gca = plt.gca()
    gca.set_yticklabels(['{:.0f}%'.format(x) for x in gca.get_yticks()]) 
    plt.legend()
    plt.savefig('output.png')  # plt.show()


def start():
    data = []
    training_data_counts = [30, 40, 50, 60, 70, 80]
    for tot in training_data_counts:
        print("-" * 80)
        print("Training data size: %d\n\n" % tot)
        data.append(batch_process(tot))
        print("\n")
    print("\nSummary:")
    print(json.dumps(data, indent=2))
    visualize(data, training_data_counts)


start()  # Let us start waiting for the result :)
