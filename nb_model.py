#TODO: Train Bayesian Classifier in this module
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#TODO: Import Cleaned Data
df = pd.read_csv("cleaned_news.csv")
print(df.head())

#TODO: split the Data
DV = "fake_news"
X = df.drop([DV], axis = 1) #independent variable
y = df[DV] #dependent variable

#TODO: Split data into training and testing portions
#Train on 75% of data, then use remaining 25% to test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

#TODO: Training data set
count_vect = CountVectorizer(max_features = 5000)# limiting to 5000 unique words, but room to play with this here!

X_train_counts = count_vect.fit_transform(X_train)
print(count_vect.vocabulary_) # here is our bag of words!
X_test = count_vect.transform(X_test) # note: we don't fit it to the model! Or else this is all useless

#TODO: Multinomial Calculations
Naive = MultinomialNB()
Naive.fit(X_train_counts, y_train)

#TODO: Predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)
# Use accuracy_score function to get the accuracy
# very accurate bc of assumption of independence!
print("Accuracy Score:", accuracy_score(predictions_NB, y_test)*100)

#TODO: real life article prediction
#link: https://entertainment.theonion.com/drake-fans-accuse-kenny-chesney-of-manipulating-billboa-1843484082
onion = ["""Calling the country singer’s place at the top of Top 200 completely illegitimate, fans of the
            rapper–singer Drake took to social media Friday to accuse Kenny Chesney of manipulating Billboard’s
            algorithm by putting effort into his album. “It’s just unfair that this guy could keep Drake from his
            rightful place on the charts by putting out quality music that he actually cares about,” said Aiden
            Howard, 14, who echoed the sentiments of Drake fans worldwide in his assertion that the artist’s
            mediocre B-sides deserved more acclaim and recognition. “He clearly gamed the streaming numbers when
            he decided to put time and energy into his craft. It’s such horseshit that Billboard rewards that
            behavior and punishes Drizzy for making a half-assed mixtape full of songs he’d already dropped on
            SoundCloud. How the hell is ‘Toosie Slide’ going to compare to a song that the artist thought about
            for more than 15 minutes?” At press time, Drake released a statement asking fans to ignore Kenny
            Chesney and focus on the horseshit that he just released."""]

onion_vec = count_vect.transform(onion) # create bag of words
predict_onion = Naive.predict(onion_vec) # applying it to the trained model
# 1: fake news!
print(predict_onion)
