import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from word_forms.word_forms import get_word_forms
from nltk import word_tokenize

outbreak_category = ['Burial practices/dead body management', 'Cases/Person(s) Under Investigation (PUI)',
                     'Cost/money/monetary donations/grants/funding', 'Death(s)',
                     'Declarations of being Ebola-Free/Recovery', 'Ebola Responder(s)',
                     'Health Care Worker(s) (HCWs)', 'Hospitals and Treatment Facilities',
                     'Infrastructure', 'Key organizations', 'Lessons learned/critiques or praise for the response',
                     'Personal Protective Equipment (PPE)', 'Preparation for potential cases',
                     'Quarantine(s)/Isolation', 'Signs/symptoms', 'Stigma', 'Suvivor(s)',
                     'Transmission', 'Travel/movement/borders/screening', 'Treatment', 'Vaccines']
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
                    #print token
                    break
    return flag

def find_existence_without_word_form(tweet_token,vocab):
    flag=False
    for token in tweet_token:
        if token == vocab:
            print vocab,token
            flag=True
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


def get_tweet_category(tweet_id, tweet_body, word_dict):
    fixed_list = [None] * 24
    tweet_token = word_tokenize(tweet_body)
    # print "In get_tweet_category"
    category_flag = False
    categories = ""
    temp_list = []
    tweet_list = []
    for key, value in word_dict.iteritems():
        flag = False
        for token in value:
            token = token.replace(".", "")
            #vocab_forms = get_word_forms(token)
            flag = find_existence_without_word_form(tweet_token, token)
            #flag=find_existense(tweet_token,vocab_forms)
            if (flag):
                break
        if (flag):
            temp_list.append(key)
    if (len(temp_list) > 0):
        # categories=",".join(temp_list)
        fixed_list[0] = tweet_id
        fixed_list[1] = tweet_body
        for cat_value in temp_list:
            index = get_index(cat_value)
            fixed_list[index + 2] = 1
            # tweet_list=[tweet_id,tweet_body,categories]

    else:
        fixed_list[0] = tweet_id
        fixed_list[1] = tweet_body
        fixed_list[len(fixed_list) - 1] = 1
        # tweet_list=[tweet_id,tweet_body,"No category"]
    return fixed_list


def replace_special_character(tweet_body):
    result = re.sub('[^a-zA-Z\n\.]', ' ', tweet_body).replace(".", "")
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    return result.strip()

def pre_process_tweet_text(tweet_text):
    text = tweet_text.lower()
    text=text.replace('ebola','')
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    temp = re.sub(r'\brt\b', '', text).strip(' ')
    temp=re.sub(r'\b\w{1,1}\b', '', temp)
    return temp


def replace_emoji(tweet_body):
    print tweet_body
    return emoji_pattern.sub(r'', tweet_body)


def read_word_libraray(location):
    df = pd.read_excel(location)
    df_new = pd.DataFrame({column: get_column_array(df, column) for column in df.columns})
    word_dict = {}
    for column in df_new:
        list_temp = df_new[column].tolist()
        str_list = filter(None, list_temp)
        str_list_temp = map(lambda x: x.lower(), str_list)
        word_dict[column] = str_list_temp
    return word_dict


cachedStopWords = stopwords.words("english")


def read_training_data(location, word_dict):
    tweet_classification = []
    df = pd.read_excel(location,)
    # df.dropna(how='any',inplace=True)
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_emoji(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_url(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: replace_special_character(x))
    df['tweet_body'] = df['tweet_body'].apply(lambda x: pre_process_tweet_text(x))
    df_new = df[['tweet_id', 'tweet_body']]
    df_temp = pd.DataFrame(df_new)
    i = 0
    for index, row in df_temp.iterrows():
        tweet_id = row['tweet_id']
        text = row['tweet_body'].lower()
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        temp = re.sub(r'\brt\b', '', text).strip(' ')

        tweet_list = get_tweet_category(tweet_id, temp, word_dict)

        tweet_classification.append(tweet_list)

    return tweet_classification


if __name__ == '__main__':
    word_dict = read_word_libraray("/home/dharamendra/Downloads/CHD_OutbreakWordLibrary.xlsx")
    print "Word libarary read completed"
    tweet_classification = read_training_data("/home/dharamendra/Downloads/tweetdata/FinalDataSet/testing.xlsx",word_dict)
    print "Tweets preprocess completed"
    # print tweet_classification
    test_df = pd.DataFrame(tweet_classification, columns=labels)
    test_df.to_csv("/home/dharamendra/Downloads/tweetdata/FinalDataSet/automated_tweets1.csv")
    print "Tweet classified"