import json
import pickle
from AggregateClassifier import AggregateClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ClassifySentiment import get_features
# with open("newPosts.json") as file:
#     data = json.load(file)

# NB_classifier = pickle.load(open("Data/Pickles/NB_classifier.pickle", "rb"))
# MNB_classifier = pickle.load(open("Data/Pickles/MNB_classifier.pickle", "rb"))
# BernoulliNB_classifier = pickle.load(open("Data/Pickles/BNB_classifier.pickle", "rb"))
# LogisticRegression_classifier = pickle.load(open("Data/Pickles/LR_classifier.pickle", "rb"))
# SGDClassifier_classifier = pickle.load(open("Data/Pickles/SGD_classifier.pickle", "rb"))
# LinearSVC_classifier = pickle.load(open("Data/Pickles/LSVC_classifier.pickle", "rb"))
# NuSVC_classifier = pickle.load(open("Data/Pickles/NuSVC_classifier.pickle", "rb"))
#
# agg_classifier = AggregateClassifier(NB_classifier, MNB_classifier,
#                                      BernoulliNB_classifier,
#                                      LogisticRegression_classifier,
#                                      SGDClassifier_classifier,
#                                      # SVC_classifier,
#                                      LinearSVC_classifier,
#                                      NuSVC_classifier)
#
# feature_list_pickle = open("Data/Pickles/feature_list.pickle", "rb")
# feature_list = pickle.load(feature_list_pickle)
# feature_list_pickle.close()
#
#
# VADER_analyzer = SentimentIntensityAnalyzer()
# n = 0
# agree = 0
# with open('delete.json', 'w') as output:
#     for post in data['data']['children']:
#         title = post['data']['title']
#         features = get_features(title, feature_list)
#
#         agg_sentiment = agg_classifier.classify(features)
#         agg_confidence = agg_classifier.recent_confidence_score()
#         if agg_confidence < 0.5:
#             agg_s = 'neu'
#         else:
#             agg_s = agg_sentiment
#
#         VADER_sentiment = VADER_analyzer.polarity_scores(title)['compound']
#         if VADER_sentiment > 0:
#             vs = 'pos'
#         elif VADER_sentiment == 0:
#             vs = 'neu'
#         else:
#             vs = 'neg'
#
#         if vs == agg_s:
#             agree += 1
#
#         json.dump({'index': n, 'content': title, 'human-score': 0,
#                   'agg_classifier-score': {'sentiment': agg_sentiment,
#                                            'confidence': agg_confidence},
#                   'VADER-score': VADER_analyzer.polarity_scores(title)}, output, indent=4)
#         n += 1
#
# print(agree/n)

import praw

reddit = praw.Reddit(client_id='vx-6S2EFdvb8mQ',
                     client_secret='3-8cSUuPh9mfrE1BgnihKxWMgyo',
                     user_agent='MacOS:http://www.testing.com/pleasework by Hennmeister')

n=0
data = {'data': []}
for submission in reddit.subreddit('technology').top(limit=1000):
    n += 1
    data['data'].append({"text": submission.title, "sentiment": ''})

with open("Data/RedditPosts/technology", 'w') as output:
    json.dump(data, output, indent=4)

print(n)
