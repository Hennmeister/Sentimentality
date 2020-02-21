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
    # data_x, data_y = get_labelled_posts(False)
    import pandas as pd
    #classifier = pickle.load(open("sentiment_classifier.pickle", "rb"))
    df = pd.read_json('dab.json', orient="split")
    # sentiments = classifier.predict_proba(df['Title'])
    # pos_sent = []
    # print("here")
    # for x in sentiments:
    #     pos_sent = x[1]
    # df["sentiement"] = pos_sent
    df['Contains Reddit'] = df['Title'].str.contains('reddit', case=False).astype(int)
    with open('dab1.json', 'w') as f:
        f.write(df.to_json(orient='split'))
