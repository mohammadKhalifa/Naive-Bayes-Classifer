import re , math
from collections import defaultdict

class NaiveBayesClassifier :
    
    def __init__(self ,classes = [True , False] ,k=0.5):
        self.k = k
        self.word_probs = []
        self.classes = classes
        
    def tokenize(self , doc):
        doc = doc.lower()
        all_words = re.findall("[a-z0-9]+" , doc)
        return set(all_words)


    def count_words(self , docs  , labels):
        counts = defaultdict(lambda : [0 , 0])
        for doc , sentiment in zip(docs , labels):
            for word in self.tokenize(doc):
                counts[word][0 if sentiment == self.classes[0] else 1] += 1
        return counts


    def words_probabilty(self , counts , total_positive , total_negative, k=1.0):
        """ returns list of tuples (word , P(X=word| Positive) , P(X=word | Negative)) """
        return [( word , (counts[word][0] + k) / (2*k + total_positive) , 
                ( (counts[word][1] + k) / (2*k + total_negative)))
                for word in counts.keys()]


    def positive_probability (self , doc , word_probs):
    
        doc_words = self.tokenize(doc)
        log_prob_positive , log_prob_negative = 0.0,0.0

        for word , pp , pn in word_probs:
            #print word , pp , pn
            if word in doc_words :
                log_prob_positive += math.log(pp)
                log_prob_negative += math.log(pn)
            else :
                log_prob_positive += math.log(1 - pp)
                log_prob_negative += math.log(1 - pn)
        
        prob_postive = math.exp(log_prob_positive)
        prob_negative = math.exp(log_prob_negative)
        if prob_negative == 0.0 and prob_postive == 0.0 :
            return 0.0
        return prob_postive / (prob_postive + prob_negative)
        
    
        
    def train(self , training_set):
        docs , labels = zip(*training_set)
        
        cnts = self.count_words(docs , labels)
        self.word_probs = self.words_probabilty(cnts , labels.count(self.classes[0]) ,
                                     labels.count(self.classes[1]) , self.k)
        
    
    def classify(self , doc):
        return self.classes[0] if self.positive_probability(doc , self.word_probs) >=0.5 else self.classes[1]
        
        