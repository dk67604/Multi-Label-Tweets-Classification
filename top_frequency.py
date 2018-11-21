import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from word_forms.word_forms import get_word_forms
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import csv

outbreak_category = ['Burial practices/dead body management', 'Cases/Person(s) Under Investigation (PUI)',
                     'Cost/money/monetary donations/grants/funding', 'Death(s)',
                     'Declarations of being Ebola-Free/Recovery', 'Ebola Responder(s)',
                     'Health Care Worker(s) (HCWs)', 'Hospitals and Treatment Facilities',
                     'Infrastructure', 'Key organizations', 'Lessons learned/critiques or praise for the response',
                     'Personal Protective Equipment (PPE)', 'Preparation for potential cases',
                     'Quarantine(s)/Isolation', 'Signs/symptoms', 'Stigma', 'Suvivor(s)',
                     'Transmission', 'Travel/movement/borders/screening', 'Treatment', 'Vaccines']
category = ["3. Burial practices/dead body management", "4. Cases/Person(s) Under Investigation (PUI)",
            "5. Cost/money/monetary donations/grants/funding", "6. Death(s)",
            "7. Declarations of being Ebola-Free/Recovery", "8. Ebola Responder(s)", "9. Health Care Worker(s) (HCWs)",
            "10. Hospitals and Treatment Facilities", "11. Infrastructure",
            "12. Key organizations", "13. Lessons learned/critiques or praise for the response",
            "14. Personal Protective Equipment (PPE)", "15. Preparation for potential cases",
            "16. Quarantine(s)/Isolation",
            "17. Signs/symptoms", "18. Stigma", "19. Suvivor(s)", "20. Transmission",
            "21. Travel/movement/borders/screening", "22. Treatment", "23. Vaccines"]
labels = ['tweet_id', 'tweet_body', 'Burial practices/dead body management',
          'Cases/Person(s) Under Investigation (PUI)', 'Cost/money/monetary donations/grants/funding', 'Death(s)',
          'Declarations of being Ebola-Free/Recovery', 'Ebola Responder(s)',
          'Health Care Worker(s) (HCWs)', 'Hospitals and Treatment Facilities',
          'Infrastructure', 'Key organizations', 'Lessons learned/critiques or praise for the response',
          'Personal Protective Equipment (PPE)', 'Preparation for potential cases',
          'Quarantine(s)/Isolation', 'Signs/symptoms', 'Stigma', 'Suvivor(s)',
          'Transmission', 'Travel/movement/borders/screening', 'Treatment', 'Vaccines', 'No Category']


def get_column_array(df, column):
    expected_length = len(df)
    current_array = df[column].dropna().values
    if len(current_array) < expected_length:
        current_array = np.append(current_array, [''] * (expected_length - len(current_array)))
    return current_array


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def replace_url(tweet_body):
    result = re.sub(r"http\S+", "", tweet_body)
    return result


def find_existense(tweet_token, vocab_forms):
    flag = False
    for token in tweet_token:
        for key, value in vocab_forms.iteritems():
            if (len(value) > 0):
                if (token in value):
                    flag = True
                    print token
                    break
    return flag


# def check_tweet_category(category_key,category_value,tweet_id_tweet_body):

def format_category(category):
    temp = category.split(".")
    return temp[1].strip(" ")


def get_index(category):
    # print category
    temp_cat = format_category(category)
    for index, val in enumerate(outbreak_category):
        if temp_cat == val:
            # print index
            return index




def replace_special_character(tweet_body):
    result = re.sub('[^a-zA-Z\n\.]', ' ', tweet_body).replace(".", "")
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    return result.strip()


def replace_emoji(tweet_body):
    return emoji_pattern.sub(r'', tweet_body)


def read_word_libraray(location):
    df = pd.read_excel(location)
    df_new = pd.DataFrame({column: get_column_array(df, column) for column in df.columns}).drop(labels='Keywords',
                                                                                                axis=1)

    word_dict = {}
    for column in df_new:
        list_temp = df_new[column].tolist()
        str_list = filter(None, list_temp)
        str_list_temp = map(lambda x: x.lower(), str_list)
        word_dict[column] = str_list_temp
    return word_dict


cachedStopWords = stopwords.words("english")

def pre_process_tweet_text(tweet_text):
    text = tweet_text.lower()
    text=text.replace('ebola','')
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    temp = re.sub(r'\brt\b', '', text).strip(' ')
    temp=re.sub(r'\b\w{1,1}\b', '', temp)
    return temp

def read_training_data(location):
    tweet_classification = []
    df = pd.read_excel(location)

    # df.dropna(how='any',inplace=True)
    k=0
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_emoji(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_url(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_special_character(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: pre_process_tweet_text(x))
    word_library={}
    for i,value in enumerate(category):

        df_temp = df.loc[df[value] == 1]

        if(len(df_temp))<=10 :
            k=3
        if 10<len(df_temp)<=50:
            k=5
        if 50 < len(df_temp) <=100:
            k=6
        if 100<len(df_temp)<=200:
            k=8
        if len(df_temp)>200:
            k=6

        countvec = CountVectorizer(max_features=k)
        count_term=pd.DataFrame(countvec.fit_transform(df_temp.tweet_body).toarray(), columns=countvec.get_feature_names())
        print value, len(df_temp)
        word_library[value]=list(count_term)

    word_library_temp={}
    for key,value in word_library.iteritems():
        temp=[None]*20
        for index,val in enumerate(value):
            temp[index]=val
        word_library_temp[key]=temp
    word_library_df=pd.DataFrame(word_library_temp)
    word_library_df.to_csv("/home/dharamendra/Downloads/tweetdata/FinalDataSet/word_library.csv")


if __name__ == '__main__':
    # word_dict = read_word_libraray("/home/dharamendra/Downloads/CHD_OutbreakWordLibrary.xlsx")
    # print "Word libarary read completed"
    tweet_classification = read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/training.xlsx")
    # print "Tweets preprocess completed"
    # print tweet_classification
    # test_df = pd.DataFrame(tweet_classification, columns=labels)
    # test_df.to_csv("EN1S4_Gp_041.csv")
    # print "Tweet classified"