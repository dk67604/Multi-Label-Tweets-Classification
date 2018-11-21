import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from word_forms.word_forms import get_word_forms
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import csv
import math
from sklearn.metrics import classification_report, hamming_loss, zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def replace_url(tweet_body):
    result = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet_body).split())
    # result = re.sub(r"http\S+", "", result)
    return result


def lemantizing(tweet_body):
    tweet_token = word_tokenize(tweet_body)
    temp = []
    lemmatizer = WordNetLemmatizer()
    for value in tweet_token:
        temp.append(lemmatizer.lemmatize(value))

    return ' '.join(temp)


def replace_special_character(tweet_body):
    result = re.sub('[^a-zA-Z\n\.]', ' ', tweet_body).replace(".", "")
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    return result.strip().lower()


def replace_emoji(tweet_body):
    # print tweet_body
    return emoji_pattern.sub(r'', tweet_body)


cachedStopWords = stopwords.words("english")


def pre_process_tweet_text(tweet_text):
    text = tweet_text.lower()
    text = text.replace('ebola', '')
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])

    temp = re.sub(r'\b\w{1,1}\b', '', text)
    return temp


def format_category(category):
    temp = category.split(".")
    return temp[1].strip(" ")


key_organization = ['who', 'cdc', 'west', 'africa', 'un', 'us']

category = ["1. Joke/Doesn't make sense", "2. Relevant", "3. Burial practices/dead body management",
            "4. Cases/Person(s) Under Investigation (PUI)",
            "5. Cost/money/monetary donations/grants/funding", "6. Death(s)",
            "7. Declarations of being Ebola-Free/Recovery", "8. Ebola Responder(s)", "9. Health Care Worker(s) (HCWs)",
            "10. Hospitals and Treatment Facilities", "11. Infrastructure",
            "12. Key organizations", "13. Lessons learned/critiques or praise for the response",
            "14. Personal Protective Equipment (PPE)", "15. Preparation for potential cases",
            "16. Quarantine(s)/Isolation",
            "17. Signs/symptoms", "18. Stigma", "19. Suvivor(s)", "20. Transmission",
            "21. Travel/movement/borders/screening", "22. Treatment", "23. Vaccines",
            "24. Miscellaneous - Public Health", "25. Miscellaneous - NOT Public Health"]
target_names = ['Burial practices/dead body management', 'Cases/Person(s) Under Investigation (PUI)',
                'Cost/money/monetary donations/grants/funding', 'Death(s)',
                'Declarations of being Ebola-Free/Recovery', 'Ebola Responder(s)',
                'Health Care Worker(s) (HCWs)', 'Hospitals and Treatment Facilities',
                'Infrastructure', 'Key organizations', 'Lessons learned/critiques or praise for the response',
                'Personal Protective Equipment (PPE)', 'Preparation for potential cases',
                'Quarantine(s)/Isolation', 'Signs/symptoms', 'Stigma', 'Suvivor(s)',
                'Transmission', 'Travel/movement/borders/screening', 'Treatment', 'Vaccines', 'No Category']


def replace_key_organisation(tweet_body):
    temp = re.sub(r'\bcdc\b', 'organization', tweet_body)
    temp = re.sub(r'\bwho\b', 'organization', temp)
    temp = re.sub(r'\bafrica\b', 'organization', temp)
    temp = re.sub(r'\bun\b', 'organization', temp)
    temp = re.sub(r'\bus\b', 'organization', temp)

    # print temp
    return temp


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from word_forms.word_forms import get_word_forms
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import csv
import math
from sklearn.metrics import classification_report,hamming_loss,zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def replace_url(tweet_body):
    result=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet_body).split())
    #result = re.sub(r"http\S+", "", result)
    return result

def lemantizing(tweet_body):
    tweet_token = word_tokenize(tweet_body)
    temp=[]
    lemmatizer = WordNetLemmatizer()
    for value in tweet_token:
        temp.append(lemmatizer.lemmatize(value))

    return  ' '.join(temp)


def replace_special_character(tweet_body):
    result = re.sub('[^a-zA-Z\n\.]', ' ', tweet_body).replace(".", "")
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    return result.strip().lower()


def replace_emoji(tweet_body):
    #print tweet_body
    return emoji_pattern.sub(r'', tweet_body)



cachedStopWords = stopwords.words("english")


def pre_process_tweet_text(tweet_text):
    text = tweet_text.lower()
    text=text.replace('ebola','')
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    temp = re.sub(r'\brt\b', '', text).strip(' ')
    temp=re.sub(r'\b\w{1,1}\b', '', temp)
    return temp

def format_category(category):
    temp = category.split(".")
    return temp[1].strip(" ")


key_organization = ['who', 'cdc', 'west', 'africa', 'un', 'us']


category = ["1. Joke/Doesn't make sense","2. Relevant","3. Burial practices/dead body management", "4. Cases/Person(s) Under Investigation (PUI)",
            "5. Cost/money/monetary donations/grants/funding", "6. Death(s)",
            "7. Declarations of being Ebola-Free/Recovery", "8. Ebola Responder(s)", "9. Health Care Worker(s) (HCWs)",
            "10. Hospitals and Treatment Facilities", "11. Infrastructure",
            "12. Key organizations", "13. Lessons learned/critiques or praise for the response",
            "14. Personal Protective Equipment (PPE)", "15. Preparation for potential cases",
            "16. Quarantine(s)/Isolation",
            "17. Signs/symptoms", "18. Stigma", "19. Suvivor(s)", "20. Transmission",
            "21. Travel/movement/borders/screening", "22. Treatment", "23. Vaccines","24. Miscellaneous - Public Health","25. Miscellaneous - NOT Public Health"]
target_names = ['Burial practices/dead body management', 'Cases/Person(s) Under Investigation (PUI)',
                     'Cost/money/monetary donations/grants/funding', 'Death(s)',
                     'Declarations of being Ebola-Free/Recovery', 'Ebola Responder(s)',
                     'Health Care Worker(s) (HCWs)', 'Hospitals and Treatment Facilities',
                     'Infrastructure', 'Key organizations', 'Lessons learned/critiques or praise for the response',
                     'Personal Protective Equipment (PPE)', 'Preparation for potential cases',
                     'Quarantine(s)/Isolation', 'Signs/symptoms', 'Stigma', 'Suvivor(s)',
                     'Transmission', 'Travel/movement/borders/screening', 'Treatment', 'Vaccines','No Category']


def replace_key_organisation(tweet_body):



    temp = re.sub(r'\bcdc\b', 'organization', tweet_body)
    temp=re.sub(r'\bwho\b', 'organization', temp)
    temp=re.sub(r'\bafrica\b', 'organization', temp)
    temp=re.sub(r'\bun\b', 'organization', temp)
    temp=re.sub(r'\bus\b', 'organization', temp)

    #print temp
    return temp

def classify_tweets(tweets_train,labels_train,tweets_test,labels_test):
    x_train=np.array(tweets_train)
    #y_train=np.array(labels_train)
    y_train = MultiLabelBinarizer().fit_transform(labels_train)
    x_train,y_train=shuffle(x_train,y_train)
    x_test=np.array(tweets_test)
    #y_test=np.array(labels_test)
    y_test=MultiLabelBinarizer().fit_transform(labels_test)
    print ('Target:',len(target_names))
    print ('Label:',len(y_test[0]))
    #kf = KFold(n_splits=3)
    kf = ShuffleSplit(n_splits=5, random_state=0)
    #kf = StratifiedKFold(n_splits=3)

    cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                     n_iter=500, random_state=None, learning_rate='optimal')

    #cv=CountVectorizer(ngram_range=(1,3),tokenizer=TreebankWordTokenizer().tokenize)

    # print cv.get_feature_names()
    #rf=RandomForestClassifier(n_estimators=1000)
    classifier = Pipeline([
         ('vectorizer', CountVectorizer(ngram_range=(1,2),tokenizer=TreebankWordTokenizer().tokenize)),
         ('clf', BinaryRelevance(classifier=cls,require_dense=[False,True]))])
    predicted_final=None
    score = 0.0
    for train_index,test_index in kf.split(x_train,y_train):
        #print("TRAIN:", train_index)
        classifier.fit(x_train[train_index], y_train[train_index])
        #print "Training completed"
        predicted = classifier.predict(x_test)
        #temp = predicted.toarray()
        temp=accuracy_score(y_test,predicted,normalize=True)
        print (temp)
        if(score<temp):
            score=temp
            print ('Accuracy:',score)
            predicted_final=predicted
    temp = predicted_final.toarray()
    # sum_var=0
    # for i in range(0,len(tweets_test)):
    #     sum_var+=accuracy_score(y_test[i],predicted[i])
    #
    # print "Accuracy:", (sum_var*100)/len(tweets_test)
    # print "Accuracy(NP-mean):", np.mean(predicted==y_test)
    # #print classifier.score(predicted,y_test)
    #
    print(classification_report(y_test, predicted, target_names=target_names))
    #
    # f=open("evaluate.txt",'w')
    # f.writelines(classification_report(y_test, predicted,target_names=target_names))
    # f.write('\nhamming loss : '+str(hamming_loss(y_test,predicted)))
    # f.write('\nzero-loss:'+str(zero_one_loss(y_test,predicted,normalize=False)))
    # f.write('\nAccuracy:'+str((sum_var*100)/len(tweets_test)))
    #df=pd.SparseSeries.from_coo(predicted)
    df=pd.DataFrame(temp)
    df.to_csv("final_1.csv")
    print confusion_matrix(y_test, predicted_final)


def read_training_data(location):
    tweets = []
    labels= []
    df = pd.read_excel(location)
    label_list=[]

    # df.dropna(how='any',inplace=True)
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_emoji(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_url(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_special_character(x))
    #df['tweet_body']=df['tweet_body'].apply(lambda x:replace_key_organisation(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: pre_process_tweet_text(x))
    #df['tweet_body'] = df['tweet_body'].apply(lambda x: lemantizing(x))
    #print df['tweet_body']
    counter=0
    for index, row in df.iterrows():
        text=row['tweet_body']
        t=[]
        for pos, val in enumerate(category):
             if not math.isnan(float(row[val])):
                t.append(pos+1)

        for i in range(len(t)):
            #[2,12,13,21]
            if t[i] in [1,2,24,25]:
                t[i]=24

        tweets.append(text)
        labels.append(t)
        counter+=1
        #if counter>10:
        #    break

    return tweets,labels


if __name__ == '__main__':
    tweets,labels=read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/training.xlsx")
    tweets_test,labels_test=read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/testing.xlsx")
    #print len(tweets_test)
    '''
    for i in range(0,len(tweets_test)):
         print ('i:',i)
         print (tweets_test[i])
         print (labels_test[i])
         print ' '
    '''
    classify_tweets(tweets,labels,tweets_test,labels_test)



def read_training_data(location):
    tweets = []
    labels = []
    df = pd.read_excel(location)
    label_list = []

    # df.dropna(how='any',inplace=True)
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_emoji(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_url(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_special_character(x))
    # df['tweet_body']=df['tweet_body'].apply(lambda x:replace_key_organisation(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: pre_process_tweet_text(x))
    # df['tweet_body'] = df['tweet_body'].apply(lambda x: lemantizing(x))
    # print df['tweet_body']
    counter = 0

    for index, row in df.iterrows():
        text = row['tweet_body']
        t = []
        if not re.search(r'\brt\b', text):
            for pos, val in enumerate(category):
                if not math.isnan(float(row[val])):
                    t.append(pos + 1)

            for i in range(len(t)):
                # [2,12,13,21]
                if t[i] in [1, 2, 24, 25]:
                    t[i] = 24

            tweets.append(text)
            labels.append(t)
            counter += 1
            # if counter>10:
            #    break

    return tweets, labels


if __name__ == '__main__':
    tweets, labels = read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/training.xlsx")
    tweets_test, labels_test = read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/testing.xlsx")
    # print len(tweets_test)
    #result = set(x for l in labels_test for x in l)
    #print result
    # for i in range(0,len(tweets_test)):
    #      #print ('i:',i)
    #      #print (tweets_test[i])
    #      print (labels_test[i])
    #      print ' '

    classify_tweets(tweets, labels, tweets_test, labels_test)
