import gensim
import matplotlib
import pandas as pd
import sys
import nltk
from docutils.nodes import inline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import logging
from gensim.models import word2vec
import math
import numpy as np
import re
import csv
from colorama import Fore
from colorama import Style
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_style("darkgrid")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

wordnet_lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def get_single_song_sentences(song, tokenizer, remove_stop_words=False, genre_bool=False, genre_name=""):
    song_sentences = tokenizer.tokenize(song.strip())
    sentences = []
    genre_type = []
    for song_sentence in song_sentences:
        # If a sentence is empty, skip it
        if len(song_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            words = re.sub(r'[^a-zA-Z\']', " ", song_sentence.lower()).split()
            if remove_stop_words:
                words = [w for w in words if not w in set(stopwords.words("english"))]
            lemmatizied_words = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in words]
            sentences.append(lemmatizied_words)
            if genre_bool:
                genre_type.append(genre_name)

    if not genre_bool:
        return sentences
    return genre_type


def parsing_data(data_set, genre=False):
    num_of_lyrics = data_set["lyrics"].size
    songs = []
    # TODO : change size
    num_of_data = 1000
    if not genre:
        for i in range(0, num_of_data):
            if ((i + 1) % 1000 == 0):
                print("Review %d of %d\n" % (i + 1, num_of_lyrics))
            if not (type(data_set["lyrics"][i]) == float):
                songs += get_single_song_sentences((data_set["lyrics"][i]), tokenizer)
    else:
        for i in range(0, num_of_data):
            if ((i + 1) % 1000 == 0):
                print("Review %d of %d\n" % (i + 1, num_of_lyrics))
            if not (type(data_set["lyrics"][i]) == float):
                songs += get_single_song_sentences((data_set["lyrics"][i]), tokenizer, False, True,
                                                   data_set["genre"][i])

    return songs


def train_model(songs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = word2vec.Word2Vec(songs, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    return model


def load_model(model_name):
    model = gensim.models.Word2Vec.load(model_name)
    # Examples for similar words
    model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
    model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    # vector = model['king'] - model['man'] + model['woman']
    # model.similar_by_vector(vector, topn=10, restrict_vocab=None)
    return model


def get_fc_data():
    temp_title = "sec\tword"
    title = [temp_title]
    fc_data = pd.read_csv('SemEval2015-English-Twitter-Lexicon.txt', names=title, header=None)
    fc_data.dropna(inplace=True)
    col_name = 'word'
    splitted_data = fc_data[temp_title].str.split("\t", n=1, expand=True)
    fc_data["score"] = splitted_data[0]
    fc_data[col_name] = splitted_data[1]
    fc_data.drop(columns=[temp_title], inplace=True)

    fc_data[col_name] = [re.sub(r'[^a-zA-Z0-9\']', "", i) for i in fc_data[col_name]]
    fc_data[col_name].replace('', np.nan, inplace=True)
    fc_data.dropna(subset=[col_name], inplace=True)
    fc_data.drop_duplicates(subset=col_name, inplace=True)
    return fc_data


def convert_word_to_vec(data, model):
    fc_data_base = pd.DataFrame()
    for single_word in data["word"]:
        try:
            fc_data_base[single_word] = model[single_word]
            # scores.append(data['score'][data['word'].tolist().index(single_word)])
            # print(single_word)
        except:
            print("'", single_word, "'", "does not appear in the model")

    return fc_data_base.transpose()


def regression_test(x, y, regressor):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    # regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    # Print the score ratio
    print(regressor.score(X_train, Y_train))
    # Loss ratio
    print(mean_squared_error(Y_test, y_pred))


def create_fc_model(x, y):
    n_in, n_h, n_out, batch_size = 300, 10, 1, 10

    X1 = torch.tensor(x.to_numpy())
    Y1 = torch.tensor(y.astype(np.float))
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.ReLU(),
                          nn.Linear(n_h, n_out),
                          nn.Tanh())

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch_size = 50
    for epoch in range(epoch_size):
        # Forward Propagation
        y_pred = model(X1)
        # Compute and print loss
        loss = loss_function(y_pred, Y1)
        print('epoch: ', epoch, ' loss: ', loss.item())
        # Zero the gradients
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()
    return model


def sentiment_analysis_part_a(w2v_model):
    # Parsing data with reges, remove duplicates, etc..
    parsed_fc_data = get_fc_data()
    # Get vector from out model for each word
    parsed_data_to_vec = convert_word_to_vec(parsed_fc_data, w2v_model)
    # Add 'score' column' and add the true label
    parsed_data_to_vec['score'] = ""
    parsed_fc_data = parsed_fc_data.set_index('word')
    for word in parsed_data_to_vec.iterrows():
        parsed_data_to_vec.at[word[0], 'score'] = (parsed_fc_data.loc[word[0]]['score'])

    y = parsed_data_to_vec["score"]
    x = parsed_data_to_vec.drop("score", axis=1)
    return x, y


def sentiment_analysis_part_b(w2v_model, fc_model, songs):
    sentiment_prediction_words = convert_word_to_vec(songs, w2v_model)
    # need to convert to tensor with ndarray
    y_pred = fc_model(torch.tensor(sentiment_prediction_words.to_numpy()))
    sentiment_prediction_words.reset_index(inplace=True)
    print(f"{Fore.GREEN}Max is \"" + sentiment_prediction_words['index'][
        int(torch.argmax(y_pred))] + "\"" + f"{Style.RESET_ALL}")

    print(f"{Fore.RED}Min is \"" + sentiment_prediction_words['index'][
        int(torch.argmin(y_pred))] + "\"" + f"{Style.RESET_ALL}")


def display_2D_results(w2v_model, most_frequent_words_by_genre):
    number_of_colors = len(most_frequent_words_by_genre)

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    print(color)

    color_list = []
    arrays = np.zeros((0, 300), dtype='f')
    for counter, (genre, words) in enumerate(most_frequent_words_by_genre.items()):
        for word in words:
            try:
                vec_word = [w2v_model[word[0]]]
                color_list.append(color[counter])
                arrays = np.append(arrays, vec_word, axis=0)
            except:
                print("'", word[0], "'", "does not appear in the model")

    reduce = PCA(n_components=50).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduce)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={'facecolors': df['color']})

    plt.xlim(Y[:, 0].min() - 10, Y[:, 0].max() + 10)
    plt.ylim(Y[:, 1].min() - 10, Y[:, 1].max() + 10)
    plt.show()


def visual_part(w2v_model, songs, dict_word_by_genre, dict_genre_by_word):
    most_frequent_words_by_genre = defaultdict(list)
    # convert to list of strings
    # flat_list = [item for sublist in songs for item in sublist]
    word_freq = defaultdict(int)
    for sent in songs['word']:
        word_freq[sent] += 1
    word_count = sorted(word_freq, key=word_freq.get, reverse=True)
    words_without_stopwords = [w for w in word_count if not w in set(stopwords.words("english"))][:3000]
    unwanted = set(dict_word_by_genre) - set(words_without_stopwords)
    for unwanted_key in unwanted: del dict_word_by_genre[unwanted_key]

    print("Print all genres for each of the most frequent words..")
    for word, genres in dict_word_by_genre.items():
        print(word + " - ", end='')
        words_occurence = sum(genres.values())
        for attribute, value in genres.items():
            print("{" + '{} : {}'.format(attribute, value) + " proportion {:.0%}".format(value / words_occurence) + "}",
                  end='')
        print("")

    for genre, words in dict_genre_by_word.items():
        most_frequent_words_by_genre[genre] = sorted(words.items(), key=lambda kv: kv[1], reverse=True)[:50]

    display_2D_results(w2v_model, most_frequent_words_by_genre)


def create_word_genre_dict(songs, songs_with_genre):
    dict_words = defaultdict(lambda: defaultdict(int))
    dict_genre = defaultdict(lambda: defaultdict(int))
    for (song, genre) in zip(songs, songs_with_genre):
        for word in song:
            dict_words[word][genre] += 1
            dict_genre[genre][word] += 1

    return dict_words, dict_genre


def clean_text(dataX, dataY):
    cleared_songs_x = []
    cleared_songs_y = []
    # TODO : change size
    for i, val in enumerate(dataX[:200]):
        # Call our function for each one, and add the result to the list of clean reviews
        if ((i + 1) % 10000 == 0):
            print("Lyrics %d of %d\n" % (i + 1, dataX.size))
        if not (type(val) == float):
            song_lyrics_by_sentences = get_single_song_sentences(val, tokenizer, remove_stop_words=True)
            flat_list = [item for sublist in song_lyrics_by_sentences for item in sublist]
            cleared_songs_x.append(" ".join(flat_list))
            cleared_songs_y.append(dataY[i])
    return cleared_songs_x, cleared_songs_y


def vectorize_features(X, vectorizer, fit=True):
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    if fit:
        features = vectorizer.fit(X)
    features = vectorizer.transform(X)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    features = features.toarray()
    return features


def text_classification_bow(cleared_songs_train, cleared_songs_test, vectorizer, cleared_y_train, cleared_y_test,
                            classes):
    X_train_bows = vectorize_features(cleared_songs_train, vectorizer)
    X_test_bows = vectorize_features(cleared_songs_test, vectorizer, fit=False)
    nbc = MultinomialNB()

    nbc.fit(X_train_bows, cleared_y_train)
    y_pred = nbc.predict(X_test_bows)
    print(confusion_matrix(cleared_y_test, y_pred))
    print(classification_report(cleared_y_test, y_pred, target_names=classes))


def get_mean_vector(word2vec_model, sentences):
    avg_vectors = []
    # remove out-of-vocabulary words
    for words in sentences:
        flat_list = [word for word in words.split() if word in word2vec_model.wv.vocab]
        if len(flat_list) >= 1:
            avg_vectors.append(np.mean(word2vec_model[flat_list], axis=0))

    return avg_vectors


def text_classification_avg_vec(w2v_model, cleared_songs_train, cleared_songs_test,cleared_y_train, cleared_y_test, classes):
    X_train_avg_vc = get_mean_vector(w2v_model, cleared_songs_train)
    X_test_avg_vc = get_mean_vector(w2v_model, cleared_songs_test)
    scaler = StandardScaler()
    scaler.fit(X_train_avg_vc)
    X_train_avg_vc_scaled = scaler.transform(X_train_avg_vc)
    # apply same transformation to test data
    X_test_avg_vc_scaled = scaler.transform(X_test_avg_vc)

    clf = MLPClassifier(hidden_layer_sizes=(50,))
    clf.fit(X_train_avg_vc_scaled, cleared_y_train)

    y_pred = clf.predict(X_test_avg_vc_scaled)
    print(confusion_matrix(cleared_y_test, y_pred))
    print(classification_report(cleared_y_test, y_pred, target_names=classes))


def text_classification_part(data_set, w2v_model):
    data_set = data_set[data_set["genre"] != "Not Available"]
    data_set = data_set[data_set["genre"].notnull()]
    data_set['genre'].value_counts().plot.bar()
    # plt.show()

    genres_to_numbers = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    genres_encoded = genres_to_numbers.fit_transform(data_set["genre"].tolist())

    # show classes to index
    classes = list(genres_to_numbers.classes_)
    # codes = genres_to_numbers.transform(list(genres_to_numbers.classes_))
    # for clss, code in zip(classes, codes):
    #     print(clss, code)

    X_train, X_test, y_train, y_test = train_test_split(data_set["lyrics"], genres_encoded, test_size=0.2,
                                                        random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    cleared_songs_train, cleared_y_train = clean_text(X_train, y_train)
    cleared_songs_test, cleared_y_test = clean_text(X_test, y_test)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=15000)

    # text_classification_bow(cleared_songs_train, cleared_songs_test, vectorizer, cleared_y_train, cleared_y_test,
    #                         classes)
    text_classification_avg_vec(w2v_model, cleared_songs_train, cleared_songs_test, cleared_y_train,cleared_y_test, classes)

    print("hello")


def main():
    # nltk.download()
    data_set = pd.read_csv("lyrics.csv")
    songs = parsing_data(data_set)
    genre_of_songs = parsing_data(data_set, genre=True)
    dict_word_by_genre, dict_genre_by_word = create_word_genre_dict(songs, genre_of_songs)

    # If we want to save :
    # flat_list = [item for sublist in songs for item in sublist]
    # np.savetxt("songs.csv", flat_list, delimiter=",", fmt='%s', header='word', comments='')
    # np.savetxt("genres.csv", songs_with_genre, delimiter=",", fmt='%s', header='word', comments='')

    # If we want to train the model again
    # model = train_model(songs)

    # If we want to load :
    songs = pd.read_csv('songs.csv')
    # songs_with_genre = pd.read_csv('file_name.csv')

    w2v_model = load_model("300features_40minwords_10context")
    # x, y = sentiment_analysis_part_a(w2v_model)
    # Test No.1 - linear regression
    print("Linear regression test has started..")
    # regression_test(x, y, LinearRegression())
    # Test No.2 - MLP regression
    print("MLP regression test has started..")
    # regression_test(x, y, MLPRegressor())
    # Test No.3 - Using fully connected
    print("Fully Connected test has started..")
    # fc_model = create_fc_model(x, y)
    # sentiment_analysis_part_b(w2v_model, fc_model, songs)
    # visual_part(w2v_model, songs, dict_word_by_genre, dict_genre_by_word)
    text_classification_part(data_set, w2v_model)
    print("hello")


if __name__ == "__main__":
    main()
