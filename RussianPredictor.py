import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import collections


def japanese_unigram_predictor(name):
    name = name.lower()
    prob_japanese = sum([unigram_prob[char] for char in name])
    return 0 if prob_japanese > 0.5 else 0

def russian_unigram_predictor(name):
    name = name.lower()
    prob_russian = sum([unigram_prob[char] for char in name])
    return 1 if prob_russian > 0.5 else 0

def spanish_unigram_predictor(name):
    name = name.lower()
    prob_spanish = sum([unigram_prob[char] for char in name])
    return 2 if prob_spanish > 0.5 else 0

def italian_unigram_predictor(name):
    name = name.lower()
    prob_italian = sum([unigram_prob[char] for char in name])
    return 3 if prob_italian > 0.5 else 0

def chinese_unigram_predictor(name):
    name = name.lower()
    prob_chinese = sum([unigram_prob[char] for char in name])
    return 4 if prob_chinese > 0.5 else 0

def czech_unigram_predictor(name):
    name = name.lower()
    prob_czech = sum([unigram_prob[char] for char in name])
    return 5 if prob_czech > 0.5 else 0

def dutch_unigram_predictor(name):
    name = name.lower()
    prob_dutch = sum([unigram_prob[char] for char in name])
    return 6 if prob_dutch > 0.5 else 0

def english_unigram_predictor(name):
    name = name.lower()
    prob_english = sum([unigram_prob[char] for char in name])
    return 7 if prob_english > 0.5 else 0

def french_unigram_predictor(name):
    name = name.lower()
    prob_french = sum([unigram_prob[char] for char in name])
    return 8 if prob_french > 0.5 else 0

def german_unigram_predictor(name):
    name = name.lower()
    prob_german = sum([unigram_prob[char] for char in name])
    return 9 if prob_german > 0.5 else 0

def greek_unigram_predictor(name):
    name = name.lower()
    prob_greek = sum([unigram_prob[char] for char in name])
    return 10 if prob_greek > 0.5 else 0

def irish_unigram_predictor(name):
    name = name.lower()
    prob_irish = sum([unigram_prob[char] for char in name])
    return 11 if prob_irish > 0.5 else 0

def vietnamese_unigram_predictor(name):
    name = name.lower()
    prob_vietnamese = sum([unigram_prob[char] for char in name])
    return 12 if prob_vietnamese > 0.5 else 0

def korean_unigram_predictor(name):
    name = name.lower()
    prob_korean = sum([unigram_prob[char] for char in name])
    return 13 if prob_korean > 0.5 else 0

def portugese_unigram_predictor(name):
    name = name.lower()
    prob_portugese = sum([unigram_prob[char] for char in name])
    return 14 if prob_portugese > 0.5 else 0

def polish_unigram_predictor(name):
    name = name.lower()
    prob_polish = sum([unigram_prob[char] for char in name])
    return 15 if prob_polish > 0.5 else 0

def arabic_unigram_predictor(name):
    name = name.lower()
    prob_arabic = sum([unigram_prob[char] for char in name])
    return 1 if prob_arabic > 0.5 else 0 


def combine_pred(unigram_probs, model_probs, weight):
    combined_probs = (1 - weight) * unigram_probs + weight * model_probs
    return combined_probs.argmax()


    

if __name__ == "__main__":

    #Linear Regression model and DataFrame for results of test.
    model = LinearRegression()
    new_set = pd.DataFrame

    #Dictionary of nationalities associated with a label(number)
    possibilities = {'Japanese': 1,
                     'Russian': 2,
                     'Spanish': 3,
                     'Italian': 4,
                     'Chinese': 5,
                     'Czech': 6, 
                     'Dutch': 7,
                     'English': 8,
                     'French': 9,
                     'German': 10,
                     'Greek': 11,
                     'Irish':12,
                     'Vietnamese':13,
                     'Korean':14,
                     'Portugese':15,
                     'Polish':16,
                     'Arabic':17,
                     'Unknown': 0}
    #reads csv
    names = pd.read_csv('surnames-test.csv')
    results = pd.read_csv('surnames-result.csv')

    #makes data frame for names column
    names['Name'] = names['Name'].str.lower()

    #makes data frame for language column
    names['Language'] = names['Language'].map(possibilities)

    #splits data into 2 sets, training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(names['Name'], names['Language'], test_size=.2, random_state=42)

    # attempting to extract features using unigram (range (1,1))
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,1))
    train_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)

    #had to do this to stop NaN error
    y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)
    y_test = np.nan_to_num(y_test, nan=0, posinf=0, neginf=0)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    #fitting linear regression model with features vector and language training set
    model.fit(train_features, y_train)
    model_pred = (model.predict(test_features > 0.5).astype(int))

    #gets count of unique char occurences
    char_counts = collections.Counter("".join(X_train))
    #gets total chars used in data
    total_chars = sum(char_counts.values())
    #probability of char occuring.  (count of char in question)/(total chars in data)
    unigram_prob = {char: count / total_chars for char, count in char_counts.items()}


    results = pd.DataFrame({'Name': X_test, 'Prediction': model_pred})
    results.to_csv('surnames-result.csv', index=False)

    accuracy = accuracy_score(y_test, model_pred)
    print(f"Accuracy: {accuracy}")

    print(classification_report(y_test, model_pred))
    # print(confusion_matrix(y_test, model_pred))

    weight = model.coef_
    print("Learned weights:", weight)

   



            