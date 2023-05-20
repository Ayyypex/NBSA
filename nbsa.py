# Import necessary libraries
import re
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Define a list of stop words to remove from the reviews
stop_words = stopwords.words('english')

# Initialize stemmer and lemmatizer objects
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):     # adjective
        return wordnet.ADJ
    elif tag.startswith('V'):   # verb
        return wordnet.VERB
    elif tag.startswith('N'):   # noun
        return wordnet.NOUN
    elif tag.startswith('R'):   # adverb
        return wordnet.ADV
    else:
        return None

# Function to lemmatize tokens
def lemmatize(tokens):
    # Part-of-speech (POS) tagging
    tagged_words = pos_tag(tokens)

    # Lemmatize each word
    lemmatized_words = []
    for word, tag in tagged_words:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:
            lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        else:
            lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemmatized_word)

    return lemmatized_words

def preprocess(review):
    """
    The preprocess function takes a review string as input and performs several text preprocessing 
    tasks to clean and normalize the text for further analysis or processing. The tasks include:

    -Removing HTML tags
    -Removing non-letter and non-whitespace characters
    -Converting all characters to lowercase
    -Tokenizing the text into individual words
    -Removing stopwords
    -Applying stemming or lemmatization
    """
    # Remove HTML tags from the review string
    review = re.sub(r'<[^>]+>', ' ', review)

    # Remove non-letter and non-whitespace characters from the review string
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)

    # Convert all characters to lowercase
    review = review.lower()

    # Tokenize text into individual words
    tokens = nltk.word_tokenize(review)

    # Remove stopwords from the token list
    tokens = [token for token in tokens if token not in stop_words]

    # Apply stemming to reduce each word to its base form
    tokens = [stemmer.stem(token) for token in tokens]

    # OR apply lemmatization
    #tokens = lemmatize(tokens)

    # Join the token list back into a string (CountVectorizer expects input documents in a string format)
    review = ' '.join(tokens)
    
    return review


class NaiveBayesClassifier:
    def __init__(self, k=1):
        """
        Initialize a Naive Bayes classifier with a k-smoothing parameter

        Parameters:
        k: The additive smoothing parameter used to avoid zero probabilities

        Attributes:
        positive_prior (float): The prior probability of the positive class
        negative_prior (float): The prior probability of the negative class
        prob_word_given_positive (numpy array): An array of shape (num_uniq_words,) containing the probability of each word given the positive class
        prob_word_given_negative (numpy array): An array of shape (num_uniq_words,) containing the probability of each word given the negative class
        """
        self.k = k
        self.positive_prior = None
        self.negative_prior = None
        self.prob_word_given_positive = None
        self.prob_word_given_negative = None

    def train(self, X, y):
        """
        Train the Naive Bayes classifier on the given training data

        Parameters:
        X (scipy.sparse matrix): A sparse matrix of shape (num_reviews, num_uniq_words) containing the bag-of-words representation of the training data
        y (numpy array): An array of shape (num_reviews,) containing the class labels for the training data
        """

        # Count number of reviews and number of unique words
        num_reviews, num_uniq_words = X.shape

        # Count number of positive reviews
        num_positive_reviews = np.sum(y == 1)

        # Calculate class priors
        self.positive_prior = num_positive_reviews / num_reviews
        self.negative_prior = 1 - self.positive_prior

        # Count the total number of occurrences of each word for each class, including k-smoothing
        positive_word_occurrences = X[y == 1].sum(axis=0) + self.k
        negative_word_occurrences = X[y == 0].sum(axis=0) + self.k

        # Calculate the total number of words for each class, including k-smoothing
        total_positive_words = np.sum(positive_word_occurrences) + self.k * num_uniq_words
        total_negative_words = np.sum(negative_word_occurrences) + self.k * num_uniq_words

        # Calculate the (posterior) probability of each word in the vocabulary given the class
        self.prob_word_given_positive = positive_word_occurrences / total_positive_words
        self.prob_word_given_negative = negative_word_occurrences / total_negative_words


    def predict(self, X):
        """
        Predict the class labels for a given feature matrix X.

        Parameters:
        X (scipy.sparse matrix): A sparse matrix of shape (num_reviews, num_uniq_words) representing the bag-of-words
        representation of the input reviews

        Returns:
        y_pred (numpy array): An array of shape (num_reviews,) containing the predicted class labels for each review in X
        """
        # Calculate log probabilities for positive and negative classes
        log_prob_positive = X @ np.log(self.prob_word_given_positive.T) + np.log(self.positive_prior)
        log_prob_negative = X @ np.log(self.prob_word_given_negative.T) + np.log(self.negative_prior)

        # Use list comprehension to assign the predicted class label based on the higher log probability
        y_pred = [1 if log_prob_positive[i] > log_prob_negative[i] else 0 for i in range(X.shape[0])]

        return np.array(y_pred)



def evaluate(classifier, X, y):
    """
    Evaluate the performance of a Naive Bayes classifier by calculating the F1 score given predicted and actual labels

    Parameters:
    classifier (object): A trained Naive Bayes classifier object with a predict method
    X (scipy.sparse matrix): A sparse matrix of shape (num_reviews, num_uniq_words) containing the input reviews 
                             to predict labels for
    y (numpy array): An array of shape (num_reviews,) containing the actual labels for the test data

    Prints:
    True Positives: Number of correct positive predictions
    False Positives: Number of incorrect positive predictions
    True Negatives: Number of correct negative predictions
    False Negatives: Number of incorrect negative predictoins
    Precision: Proportion of true positives out of all instances predicted as positive
    Recall: Proportion of true positives out of all actual positive instances
    Accuracy: Proportion of correct predictions out of all instances
    F1 measure: Harmonic mean of precision and recall
    """

    # Predict labels for the test set
    y_pred = classifier.predict(X)

    # Use list comprehension to calculate true/false positive/negative
    tp = sum([1 for i in range(len(y)) if y[i] == 1 and y_pred[i] == 1])
    fp = sum([1 for i in range(len(y)) if y[i] == 0 and y_pred[i] == 1])
    tn = sum([1 for i in range(len(y)) if y[i] == 0 and y_pred[i] == 0])
    fn = sum([1 for i in range(len(y)) if y[i] == 1 and y_pred[i] == 0])

    # Calculate precision, recall, and accuracy
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # Calculate F1 measure 
    beta = 1
    f1 = ((pow(beta, 2) + 1) * precision * recall) / (pow(beta, 2) * precision + recall)

    # Print results 
    print(f"True Positives : {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives : {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy       : {accuracy}")
    print(f"Precision      : {precision}")
    print(f"Recall         : {recall}")
    print(f"F1-measure     : {f1}")



####################################

# Initialize LabelEncoder and CountVectorizer objects
label_encoder = LabelEncoder()
vectorizer = CountVectorizer(preprocessor=preprocess)                 # bag-of-words frequency
#vectorizer = CountVectorizer(preprocessor=preprocess, binary=True)   # bag-of-words presence

# Start a timer to measure code execution time
import time
start = time.time()

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("imdb_master.csv", encoding="ISO-8859-1")

# Remove the unsupervised reviews
df = df[df["label"] != "unsup"]

# Separate the reviews and labels based on whether they are for training or testing
train_df = df[df["type"] == "train"]
test_df = df[df["type"] == "test"]
train_reviews = train_df["review"].tolist()
train_labels = train_df["label"].tolist()
test_reviews = test_df["review"].tolist()
test_labels = test_df["label"].tolist()

# Use the vectorizer to transform the training and testing reviews into a bag-of-words representation
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

# Use the label_encoder to encode pos and neg into numerical form. Encodes alphabetically, so neg=0 and pos=1
y_train = label_encoder.fit_transform(train_labels) 
y_test = label_encoder.transform(test_labels)

# Create and train the Naive Bayes classifier
nb = NaiveBayesClassifier()
nb.train(X_train, y_train)

# Evaluate the classifier
evaluate(nb, X_test, y_test)
print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")

# Print the code execution time
end = time.time()
print(f"\nProgram time: {end-start} seconds")
