import csv

def split_data():
    with open('Data/TwitterSentimentData.csv', encoding="ISO-8859-1") as input_file:
        with open('Data/negTweets', 'w') as output_neg:
            with open('Data/posTweets', 'w') as output_pos:
                data = csv.reader(input_file)
                third_i = 0
                neg_i = 0
                pos_i = 0
                for tweet_info in data:
                    if third_i % 3 == 0:
                        tweet = tweet_info[5]
                        if tweet_info[0] == '0' and neg_i < 200000:
                            output_neg.write(tweet + '\n')
                            neg_i += 1
                        elif tweet_info[0] == '4' and pos_i < 200000:
                            output_pos.writelines(tweet + '\n')
                            pos_i += 1
                    third_i += 1



if __name__ == '__main__':
    split_data()


