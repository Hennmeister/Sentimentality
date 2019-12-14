from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json


with open('redditPosts.json') as input_file:
    data = json.load(input_file)
    parsed = []
    for post in data['data']['children']:
        parsed.append(post['data']['title'])

test_text = 'It is good'

analyzer = SentimentIntensityAnalyzer()
ss = analyzer.polarity_scores(test_text)
print(ss)

for title in parsed:
    vs = analyzer.polarity_scores(title)
    print(title, str(vs))
    print()
