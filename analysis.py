import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('twitter_samples')
nltk.download('omw-1.4')


# initialize
stop_words = stopwords.words('english')

# method that takes several words in tokens array and removes stopwords and uneeded characters
def remove_noise_and_lemmatize(tokens):

    cleaned_tokens = []

    for token, part_of_speech_tag in pos_tag(tokens):
        # remove links
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        # remove mentions
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
       
        # based on the position tag, assign the wordnet equivelant
        if part_of_speech_tag.startswith('J'):
            wordnet_part_of_speech_tag = wordnet.ADJ
        elif part_of_speech_tag.startswith('V'):
            wordnet_part_of_speech_tag = wordnet.VERB
        elif part_of_speech_tag.startswith('N'):
            wordnet_part_of_speech_tag = wordnet.NOUN
        else:
            wordnet_part_of_speech_tag = wordnet.ADV

        # lemmatized the token based on the part of speech (get base of the token)
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, wordnet_part_of_speech_tag)
       
        # If the lemmatized token is not empty, it's not a punctuation and it's not a stop word, add it to the resulting array
        if len(token) > 0 and token.lower() not in stop_words and token not in string.punctuation:
            cleaned_tokens.append(token.lower())
           
    return cleaned_tokens

# reformat data so that we can later use to train Naive Bayes
def get_tweets_for_model(cleaned_tokens_list, status):
    tweet_tokens_for_model = []
    for tweet_tokens in cleaned_tokens_list:
        tweet_tokens_for_model_dict = {}
        for token in tweet_tokens:
            tweet_tokens_for_model_dict[token] = True
        tweet_tokens_for_model.append((tweet_tokens_for_model_dict, status))
    return tweet_tokens_for_model

# Train data
def train_model():
    # get positive and negative tweets for training data
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    # tokenize the positive and negative tweets
    positive_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')

    # remove noise and lemmatize
    positive_cleaned_tweets_tokens = []
    negative_cleaned_tweets_tokens = []

    for tokens in positive_tweets_tokens:
        positive_cleaned_tweets_tokens.append(remove_noise_and_lemmatize(tokens))

    for tokens in negative_tweets_tokens:
        negative_cleaned_tweets_tokens.append(remove_noise_and_lemmatize(tokens))

    # reformat data for the NaiveBayesClassifier
    positive_data = get_tweets_for_model(positive_cleaned_tweets_tokens, 'Positive')
    negative_data = get_tweets_for_model(negative_cleaned_tweets_tokens, 'Negative')

    # combine the two datasets
    data = positive_data + negative_data

    # randomly shuffle dataset so it's not just positive then negative
    random.shuffle(data)

    # split into train and test data to see the accuracy
    train_data = data[:7000]
    test_data = data[7000:]

    # train using the NaiveBayesClassifier model on the train_data
    # downside with naive bayes is that it doesn't care for the order of words
    classifier = NaiveBayesClassifier.train(train_data)

    # print accuracy of the training
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))
   
    return classifier

classifier = train_model()

def classify_tweet(custom_tweet):
    global classifier

    # if classifier is None:
    #     classifier = train_model()
   
    custom_tokens = remove_noise_and_lemmatize(word_tokenize(custom_tweet))

    #print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))

    result = classifier.classify(dict([token, True] for token in custom_tokens))

    return result
