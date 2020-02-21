import json
import praw

reddit = praw.Reddit(client_id='vx-6S2EFdvb8mQ',
                     client_secret='3-8cSUuPh9mfrE1BgnihKxWMgyo',
                     user_agent='MacOS:http://www.testing.com/pleasework by Hennmeister')

n = 0
data = {'data': []}
for submission in reddit.subreddit('askReddit').top(limit=1000):
    n += 1
    data['data'].append({"text": submission.title, "binary-sentiment": '',
                         "neutral": ''})

with open("Data/RedditPosts/askReddit", 'w') as output:
    json.dump(data, output, indent=4)


print(n)
