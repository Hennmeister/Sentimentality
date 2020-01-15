from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

def train_classifier():
    vectorizer = CountVectorizer(analyzer="word", \
                                 max_df = 0.9, \
                                 min_df = 2, \
                                 max_features=5000,
                                 )

    data_x = []
    data_y = []
    with open('Data/Tweets/cleaned_negTweets', encoding="ISO-8859-1", mode='r') as output_neg:
        line = output_neg.readline().rstrip()
        while line:
            data_x.append(line)
            data_y.append(0)
            line = output_neg.readline()
    with open('Data/Tweets/cleaned_posTweets', mode='r', encoding="ISO-8859-1") as output_pos:
        line = output_pos.readline()
        while line:
            data_x.append(line)
            data_y.append(1)
            line = output_pos.readline()

    data_x_train, data_x_test, data_y_train, data_y_test = \
        train_test_split(data_x, data_y, train_size=(0.8), random_state=14)

    data_x_train = vectorizer.fit_transform(data_x_train)
    data_x_test = vectorizer.transform(data_x_test)
    vectorizer.vocabulary_.get('good')

    print('datax_train shape', data_x_train.shape)
    data_x_train = SelectKBest(chi2, k='all').fit_transform(data_x_train, data_y_train)
    data_x_train = TfidfTransformer().fit_transform(data_x_train)
    print('datax_train shape', data_x_train.shape)

    clf = VotingClassifier([('lsvc', CalibratedClassifierCV(LinearSVC(max_iter=10000))),
                            ('mnb', MultinomialNB()),
                            ('r', RandomForestClassifier(n_estimators=100)),
                           ('lgrg', LogisticRegressionCV(random_state=1)),
                           ('bnb', BernoulliNB())], voting='soft')

    clf = clf.fit(data_x_train, data_y_train)
    print('clf score', clf.score(data_x_test, data_y_test))

    y_predicted = clf.predict(data_x_test)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_predicted, data_y_test))

    from sklearn.metrics import classification_report
    print(classification_report(data_y_test, y_predicted))

if __name__ == '__main__':
    train_classifier()
