from NativeBayesClassifier import NaiveBayesClassifier

# reading data 
f1 = open('fb_data.txt' , 'r')
f2 = open('fb_label.txt' , 'r')

comments , labels = f1.read().split('\n') , f2.read().replace('\n\n','\n').split('\n')
labels = labels[:-1]


##Extracting comments with only 'P' or 'N' sentiment

new_comments  = []
new_labels = []
for i,com in enumerate(labels):
    if labels[i] !='O':
        new_comments.append(comments[i])

new_labels = filter(lambda x: x!='O' , labels)
new_labels = map(lambda x : True if x=='P' else False , new_labels )


BC = NaiveBayesClassifier(k=1.0)

## training 
BC.train(zip (new_comments , new_labels))

#calculating training accuracy
accuracy = 0
for i , c in enumerate(new_comments):
    #print BC.classify(new_comments[i])
    if BC.classify(new_comments[i]) == new_labels[i]:
        accuracy +=1

accuracy *= 100.0 / len(new_comments)

print accuracy