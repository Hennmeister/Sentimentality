import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from config import NUM_TWEETS

allowed_word_types = ["J", "R"]
# allowed_word_types = ["J", "R", "V", "N"]

def prepare_data():
    with open('Data/Tweets/TwitterSentimentData.csv', encoding="ISO-8859-1") as input_file:
        with open('Data/Tweets/cleaned_negTweets', 'w') as output_neg:
            with open('Data/Tweets/cleaned_posTweets', 'w') as output_pos:
                data = csv.reader(input_file)
                third_i = 0
                neg_i = 0
                pos_i = 0
                for tweet_info in data:
                    if third_i % 3 == 0:
                        tweet = clean_tweet(tweet_info[5])
                        if tweet:
                            if tweet_info[0] == '0' and neg_i < NUM_TWEETS//2:
                                output_neg.write(tweet + '\n')
                                neg_i += 1
                            elif tweet_info[0] == '4' and pos_i < NUM_TWEETS//2:
                                output_pos.writelines(tweet + '\n')
                                pos_i += 1
                    third_i += 1

def clean_tweet(tweet) -> str:
    # Remove punctuation and words that start with given punctuation
    # Relevant for the @TwitterHandles - Zike doesn't remove those,
    # shouldn't be an issue cause they wont be part of top x frequencies
    tweet = tweet.lower()  # Convert to lower case
    translator = str.maketrans('', '', string.punctuation)
    tweet = re.sub("@.+?&", "", tweet.translate(translator))

    # Tokenize and remove stopwords
    stop_words = list(set(stopwords.words('english')))
    tokenized_tweet = [w for w in word_tokenize(tweet) if w not in stop_words]
    # POS tagging - only looking at adjectives for now
    pos = nltk.pos_tag(tokenized_tweet)
    cleaned_tweet = []
    for w in pos:
        # Checking if the first letter in tag is J,or R,
        # signifying adjective or adverb
        if w[1][0] in allowed_word_types:
            cleaned_tweet.append(w[0].lower())
    return ' '.join(cleaned_tweet)

if __name__ == '__main__':
    prepare_data()


