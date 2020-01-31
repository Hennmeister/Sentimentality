import json
import pickle

def get_labelled_posts(neu: bool) -> (list, list):
    with open('Data/RedditPosts/askReddit') as reddit_data:
        data = json.load(reddit_data)['data']
        data_x = []
        data_y = []
        for post in data:
            if post['binary-sentiment'] == '':
                break
            if neu and post['neutral'] == 'T':
                continue
            data_x.append(post['text'])
            pos = post['binary-sentiment'] == 'P'
            data_y.append((0, 1)[pos])
        return data_x, data_y

if __name__ == '__main__':
    data_x, data_y = get_labelled_posts(False)
    import time
    start = time.time()
    classifier = pickle.load(open("sentiment_classifier.pickle", "rb"))
    print(time.time() - start)
    for x in data_x:
        print(x, classifier.predict_proba([x]))
   # print(classifier.score(data_x, data_y))
   # print(classifier.predict_proba(["Hitler"]))
