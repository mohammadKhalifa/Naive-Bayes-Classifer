from NativeBayesClassifier import NaiveBayesClassifier
import pandas as pd 

data = pd.read_csv('amazon_reviews.csv')

data.dropna(axis = 0 , inplace=True)

docs , labels = data['review'].tolist() , data['Positive'].tolist()


BC = NaiveBayesClassifier(k=50.0 , classes=[1 , 0])

BC.train(zip (docs ,labels))

accuracy = 0
for i , c in enumerate(docs):
    if BC.classify(new_comments[i]) == labels[i]:
        accuracy +=1

accuracy *= 100.0 / len(docs)

print accuracy

