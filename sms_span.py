import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/spam.csv')
print(df.sample(5))
df.shape
df.info()
df.isnull().sum()
df.drop(columns=df.iloc[:,[2,3,4]],inplace=True)
df
df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
df
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Target'] = encoder.fit_transform(df['Target']) #converts ham to 0 and spam to 1
df
df.isnull().sum()
df.duplicated().sum() #checking duplicate values
df = df.drop_duplicates(keep='first')
df.duplicated().sum()
df.shape
df['Target'].value_counts()
plt.pie(df['Target'].value_counts(), labels = ['ham','spam'],autopct = '%0.2f')
plt.show()
#DATA IS IMBALANCED AS SHOWN IN THIS FIGURE ABOVE
import nltk
nltk.download('punkt')
df['num_char'] = df['Text'].apply(len)
df.head()
#fetching the number of words
df['num_words'] = df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.sample(5)
df['num_sentences'] = df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.describe()
# on average 78 characters are used, roughly 2 sentences are used and  max 220 wwords are used with 38 sentences
###ham
df[df['Target'] == 0] [['num_char','num_words','num_sentences']].describe()
#spam
df[df['Target'] == 1] [['num_char','num_words','num_sentences']].describe()
#137.891271 spam per messages this can show that spam message is lengthier or larger than the ham
#Character_based
plt.figure(figsize=(14,10))
sns.histplot(df[df['Target']==0]['num_char'], color ='blue')
sns.histplot(df[df['Target']==1]['num_char'],color = 'red')
#maximum ham average characters are lesser than the max spam characters clearly visible
#word_based
plt.figure(figsize=(14,10))
sns.histplot(df[df['Target']==0]['num_words'], color ='blue')
sns.histplot(df[df['Target']==1]['num_words'],color = 'red')
sns.pairplot(df, hue='Target')
#Outliers are present that can affect analysis
correlation_matrix = df[['Target', 'num_char', 'num_words', 'num_sentences']].corr()
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
string.punctuation
###Porter Stemmer for stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():#alphanumeric
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
#from nltk.corpus import stopwords
#stopwords.words('english')
#e removes as it is basics of nltk but word mean it danc means dance
transform_text("I am trying to learn Machine Learning from youtube watching the lectures. How about you? ")
df['Text'][10]
df['Transformed_text']=df['Text'].apply(transform_text)
df.head()
from wordcloud import WordCloud
wc = WordCloud(width=500, height = 500, min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['Target']==1]['Transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(14,10))
plt.imshow(spam_wc)
ham_wc = wc.generate(df[df['Target']==0]['Transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(14,10))
plt.imshow(ham_wc)
spam_msg=[]
for msg in df[df['Target']==1]['Transformed_text'].tolist():
    for word in msg.split():
        spam_msg.append(word)
len(spam_msg)
from collections import Counter
pd.DataFrame(Counter(spam_msg).most_common(30))
ham_msg=[]
for msg in df[df['Target']==0]['Transformed_text'].tolist():
    for word in msg.split():
        ham_msg.append(word)
len(ham_msg)
from collections import Counter
pd.DataFrame(Counter(ham_msg).most_common(30))
#naive bayes has the most efficient results on terms of the textual data
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['Transformed_text']).toarray()
X.shape
y = df['Target'].values
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
###Naive bayes best for text so fitting with its variants
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test,y_pred1))
print("Confusion matrix: ", confusion_matrix(y_test,y_pred1))
print("precision score: ", precision_score(y_test,y_pred1))
#bad performance with precision score and imbalanced dataset require more precision score
bnb.fit(X_train,y_train)
y_pred2 = bnb.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test,y_pred2))
print("Confusion matrix: ", confusion_matrix(y_test,y_pred2))
print("precision score: ", precision_score(y_test,y_pred2))
model=mnb.fit(X_train,y_train)
y_pred3 = mnb.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test,y_pred3))
print("Confusion matrix: ", confusion_matrix(y_test,y_pred3))
print("precision score: ", precision_score(y_test,y_pred3))
input_text = "hello you have been rewarded with a prize worth 10000 call for reward"
input_text = tfidf.transform([input_text])
# Make a prediction
prediction = model.predict(input_text)
# Check the prediction
if prediction[0] == 1:
    print("This is a spam message.")
else:
    print("This is a ham (non-spam) message.")
