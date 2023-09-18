import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
import collections


def combine_pred(unigram_probs, model_probs, weight):
    combined_probs = (weight * unigram_probs) + (weight * model_probs)
    return combined_probs

if __name__ == "__main__":

    #Linear Regression model and DataFrame for results of test.
    model = LinearRegression()
    new_set = pd.DataFrame

    #Dictionary of nationalities associated with a label(number)
    possibilities = {'Russian':1,
                     'English': 2}
    # reads csv
    names = pd.read_csv('russian-english-dev.txt', encoding='utf-8', sep=',')

    #  makes data frame for names column
    names['Name'] = names['Name'].str.lower()

    # makes data frame for language column
    names['Language'] = names['Language'].map(possibilities)

    # splits data into 2 sets, training and testing sets, set test size to 20% of data and 80% is training set
    X_train, X_test, y_train, y_test = train_test_split(names['Name'], names['Language'], test_size=.2, random_state=12)

    # attempting to extract features using unigram (range (1,1) or bigram range (2,2))
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,2))
    train_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)

    
    # fitting linear regression model with training set data
    model.fit(train_features, y_train)
    # using the model to predict labels for test set
    model_pred = model.predict(test_features > 0.5).astype(int)


    # want to make a dictionary that will assign a dict of char : probabilities from vectorizer and assigns to nationality number as key -> name_unigram_models[2] = char probs for russian labeled data
    name_unigram_models = {}

    for x in range(3):
        lang_num = names[names['Language'] == x]
        char_counts = collections.Counter("".join(lang_num['Name']))
        total_chars = sum(char_counts.values())
        unigram_probs = {char: count / total_chars for char, count in char_counts.items()}
        
        name_unigram_models[x] = unigram_probs

    #### this is to predict only using the probability of russian names with the russian character set
    unigram_pred = []

    for name in X_test:
        choices = []

        for x in range(2):
            unigram_probs = []
            unigram_probs = name_unigram_models[x]

            for char in name:
                if char not in name_unigram_models[x]:
                    unigram_probs[char] = 0.0
            
            possible_nationality = sum([unigram_probs[char] for char in name])
            
            choices.append(possible_nationality)

        
        predicted_nationality = choices.index(max(choices))
        
        unigram_pred.append(predicted_nationality)
        
        # combine predictions  
        final_predictions = [combine_pred(unigram_pred, model_pred, weight=0.5) for unigram_pred, model_pred in zip(unigram_pred, model_pred)]
        final_predictions = np.round(final_predictions).astype(int)
        final_predictions = np.abs(final_predictions)

    
    # writing results to csv
    results = pd.DataFrame({'Name': X_test, 'Prediction': final_predictions})
    results.to_csv('ru-vs-english-result.csv', index=False)

    # prints accuracy score using the test data and model predictions.
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Accuracy: {accuracy}")

    # prints classification report that shows my test data and the predictions made by the linear regression model.
    print(classification_report(y_test, final_predictions))


    # checking weights learned and f1_score but i am getting alot of negatives
    weight = model.coef_
    print("Learned weights:", weight)
    f1 = f1_score(y_test, final_predictions, average='weighted')
    print(f1)

#### printing bigrams for english names and their probabilities 

    target = 2  # change this value to choose which language you want to filter bigrams for

    mask = np.array(y_train) == target
    bigram_sums = train_features[mask].sum(axis=0)

    # gets the list of bigrams associated with the target label
    bigrams_list = vectorizer.get_feature_names_out()

    # dictionary for bigram probs
    bigram_probabilities = {}

    # go through bigrams and probs
    for i in range(len(bigrams_list)):
        bigram = bigrams_list[i]
        bigram_sum = bigram_sums[0, i]
        if bigram_sum > 0:
            bigram_probabilities[bigram] = bigram_sum

    # print the bigrams and the probability 
    for bigram, probability in bigram_probabilities.items():
        print(f"Bigram: {bigram}, Probability: {probability}")