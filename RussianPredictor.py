import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import collections

def russian_unigram_predictor(name):
    name = name.lower()
    prob_russian = sum([unigram_prob[char] for char in name])
    return 1 if prob_russian > 0.5 else 0


def combine_pred(unigram_probs, model_probs, weight):
    combined_probs = (1 - weight) * unigram_probs + weight * model_probs
    return combined_probs.argmax()

    

if __name__ == "__main__":

    #Linear Regression model and DataFrame for results of test.
    model = LinearRegression()
    new_set = pd.DataFrame

    #Dictionary of nationalities associated with a label(number)
    possibilities = {"Japanese": 0,
                     "Russian": 1,
                     "Spanish": 2,
                     "Italian": 3,
                     "Chinese": 4,
                     "Czech": 5, 
                     "Dutch": 6,
                     "English": 7,
                     "French": 8,
                     "German": 9,
                     "Greek": 10,
                     "Irish":11,
                     "Vietnamese":12,
                     "Korean":13,
                     "Portugese":14,
                     "Polish":15,
                     "Arabic":16,}
    #reads csv
    names = pd.read_csv('surnames-test.csv')

    #makes data frame for names column
    names['Name'] = names['Name'].str.lower()

    #makes data frame for language column
    names['Language'] = names['Language'].apply(lambda x: possibilities[x] if x in possibilities else 17)
    

    #splits data into 2 sets, training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(names['Name'], names['Language'], test_size=.2, random_state=42)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
    X = vectorizer.fit_transform(names['Name'])
    vectorizer.get_feature_names_out
    
    model.fit(X, names['Language'])

    features = vectorizer.transform(X_train)

    new_set['Language'] = model.predict(features)

    new_set = new_set['Name'] = X_test
    
    print(new_set['Name'] + new_set['Language'] + "\n")


    # y_pred = model.predict(X_test)

    # threshold = 0.5 
    # predicted = [1 if pred > threshold else 0 for pred in y_pred]

    # weights = model.coef_
    # print("coefficients: /n", weights)
    

    # #gets count of unique char occurences
    # char_counts = collections.Counter("".join(X_train))
    
    # #gets total chars used in data
    # total_chars = sum(char_counts.values())

    # #probability of char occuring.  (count of char in question)/(total chars in data)
    # unigram_prob = {char: count / total_chars for char, count in char_counts.items()}
    
    # y_pred = [russian_unigram_predictor(name) for name in X_test]
    

    # names.to_csv('surnames-result.csv')

    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # for name in X_test:
    #     names['Name'] = name
    #     names['Language'] = y_pred



            