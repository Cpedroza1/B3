import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import collections

def russian_unigram_predictor(name):
    name = name.lower()
    prob_russian = sum([unigram_prob[char] for char in name])
    return 1 if prob_russian > 0.5 else 0

def learn_language(language):
    if language == "Japanese":
        return 0
    elif language == "Russian":
        return 1
    # elif language == "Spanish":
    #     return 2
    # elif language == "Italian":
    #     return 3
    # elif language == "Chinese":
    #     return 4
    # elif language == "Czech":
    #     return 5
    # elif language == "Dutch":
    #     return 6
    # elif language == "English":
    #     return 7
    # elif language == "French":
    #     return 8
    # elif language == "German":
    #     return 9
    # elif language == "Greek":
    #     return 10
    # elif language == "Irish":
    #     return 11
    # elif language == "Vietnamese":
    #     return 12
    # elif language == "Korean":
    #     return 13
    # elif language == "Portugese":
    #     return 14
    # elif language == "Polish":
    #     return 15
    # elif language == "Arabic":
    #     return 16
    

if __name__ == "__main__":

    #reads csv
    names = pd.read_csv('surnames-test.csv')
    #makes data frame for names column
    names['Name'] = names['Name'].str.lower()
    #makes data frame for language column
    names['Language'] = names['Language'].apply(lambda lang: 1 if lang == "Russian" else 0)

    #splits data into 2 sets, training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(names['Name'], names['Language'], test_size=.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    weights = model.coef_
    print("coefficients: /n", weights)
    

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



            