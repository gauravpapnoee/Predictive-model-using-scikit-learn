# So here we are going to create a pridictive model using scikit learn.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample spam dataset
texts = ["Congratulations, you've won a prize!", "Call me now", "Win big cash", 
         "Hello, how are you?", "Let's meet tomorrow", "Free entry in contest"]
labels = [1, 0, 1, 0, 0, 1]  # 1 = spam, 0 = not spam

# Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels