import math
from operator import itemgetter
import random
import string

import numpy as np

testFile = open("test.txt", "r")
trainFile = open("train.txt", "r")
hamNr = 0
hamAllWordsNr = 0
spamNr = 0
spamAllWordsNr = 0
hamDict = {}
spamDict = {}
stopWords = []
allWordsNr = 0
pSpamDict = {}
pHamDict = {}
wordsNrDict = {}

def readEmail(file, dict, type, a):
    global allWordsNr

    email = open(file, "r", encoding="Latin-1")
    for line in email:
        words = line.split()
        words = words[1:]
        words = [w.lower() for w in words if w not in stopWords and w not in string.punctuation]
        for w in words:
            if type == 'init':
                if w not in hamDict and w not in spamDict:
                    allWordsNr += 1
                if w in dict:
                    dict[w] += 1
                elif w not in dict:
                    dict[w] = 1
            elif type == 'train':
                # new word -> set count, probability of ham and spam
                if w not in dict:
                    dict[w] = 1
                    if w not in spamDict:
                        pSpamDict[w] = 0.00000001
                    else:
                        pSpamDict[w] = spamDict[w]/spamAllWordsNr
                    
                    if w not in hamDict:
                        pHamDict[w] = 0.00000001
                    else:
                        pHamDict[w] = hamDict[w]/hamAllWordsNr
                else:
                    dict[w] += 1
            elif type == 'additiv':
                if w not in dict:
                    dict[w] = 1
                    if w not in spamDict:
                        pSpamDict[w] = 0.00000001
                    else:
                        pSpamDict[w] = (spamDict[w] + a)/(spamAllWordsNr + a * allWordsNr)
                    
                    if w not in hamDict:
                        pHamDict[w] = 0.00000001
                    else:
                        pHamDict[w] = (hamDict[w] + a)/(hamAllWordsNr + a * allWordsNr)
                else:
                    dict[w] += 1

for line in trainFile:
    line = line.strip()
    if 'ham' in line:
        hamNr+=1
        readEmail("enron6/ham/"+line, hamDict, 'init', 0)
    elif 'spam' in line:
        spamNr+=1
        readEmail("enron6/spam/"+line, spamDict, 'init', 0)

hamAllWordsNr = sum(hamDict.values())
spamAllWordsNr = sum(spamDict.values())

with open("stopwords.txt", "r") as f:
    text = f.read()
    text = text.split()
    for word in text:
        stopWords.append(word)

with open("stopwords2.txt", "r") as f:
    text = f.read()
    text = text.split()
    for word in text:
        stopWords.append(word)


# additiveType = train/additiv
def naivBaies(fileName, type, additiveType, a):
    # init
    hamTestNr = 0
    spamTestNr = 0
    falseNeg = 0
    falsePoz = 0
    nr = 0
    f = open(fileName, 'r')

    # train 
    for line in f:
        wordsNrDict.clear()
        pHamDict.clear()
        pSpamDict.clear() 
        nr+=1
        line = line.strip()
        if 'ham' in line:
            hamTestNr+=1
            readEmail("enron6/ham/"+line, wordsNrDict, additiveType, a)
        elif 'spam' in line:
            spamTestNr+=1
            readEmail("enron6/spam/"+line, wordsNrDict, additiveType, a)
        # classification
        sum = 0
        for word in wordsNrDict:
            sum += wordsNrDict[word] * (math.log(pSpamDict[word]) - math.log(pHamDict[word]))
        # lnR > 0 --> spam      lnR < 0 --> ham
        lnR = math.log(spamNr/(spamNr+hamNr)) - math.log(hamNr/(spamNr+hamNr)) + sum
        # check
        if  lnR < 0 and 'spam' in line:
            falseNeg += 1
        elif  lnR > 0 and 'ham' in line:
            falsePoz += 1

    print("----------------------------------------", type , "-------- alpha = ", a)
    print("Mistakes: ", (falseNeg+falsePoz)*100/nr, " %")
    print("False pozitive: ", (falsePoz*100)/hamTestNr, ' %')
    print("False negative: ", (falseNeg*100)/spamTestNr, " %")


def crossValidation(file, k):
    global hamNr, spamNr
    allEmails = []
    allErrors = []
    f = open(file, 'r')
    for line in f: 
        allEmails.append(line.strip())

    for a in [0.00005, 0.0005, 0.005, 0.5, 1]:
        hamNr = 0 
        spamNr = 0
        hamTestNr = 0
        spamTestNr = 0
        spamDict = {}
        hamDict = {}

        random.shuffle(allEmails)
        all_split_np = np.array_split(np.array(allEmails), k)
        all_split = [x.tolist() for x in all_split_np]
        # i. chunk --> test
        for i in range(k):
            train = []
            test = all_split[i]
            for j in range(k):
                if i != j:
                    train += all_split[j]
        
        # train 
        for email_name in train:
            if 'ham' in email_name:
                hamNr+=1
                readEmail("enron6/ham/"+email_name, hamDict, 'init', 0)
            elif 'spam' in email_name:
                spamNr+=1
                readEmail("enron6/spam/"+email_name, spamDict, 'init', 0)

        mistake = 0
        # test
        for line in test: 
            pHamDict.clear()
            pSpamDict.clear()
            wordsNrDict.clear()
            if 'ham' in line:
                hamTestNr+=1
                readEmail("enron6/ham/"+line, wordsNrDict, 'additiv', a)
            elif 'spam' in line:
                spamTestNr+=1
                readEmail("enron6/spam/"+line, wordsNrDict, 'additiv', a)

           # classification
            sum = 0
            for word in wordsNrDict:
                sum += wordsNrDict[word] * (math.log(pSpamDict[word]) - math.log(pHamDict[word]))
            # lnR > 0 --> spam      lnR < 0 --> ham
            lnR = math.log(spamNr/(spamNr+hamNr)) - math.log(hamNr/(spamNr+hamNr)) + sum
            # check
            if  lnR < 0 and 'spam' in line:
                mistake += 1
            elif  lnR > 0 and 'ham' in line:
                mistake += 1
        allErrors.append((a,(mistake*100)/(hamTestNr+spamTestNr)))
    
    sorted(allErrors, key=itemgetter(1))
    for (a, value) in allErrors:
        print("alpha = ", a, "  error = ", value)
 
def HalfSupervised():
    added = True 
    
    fileNr = [*range(1000)]
    while added:
        hamNr = 0 
        spamNr = 0
        # init
        trainFile = open("train.txt", "r")
        for line in trainFile:
            line = line.strip()
            if 'ham' in line:
                hamNr+=1
                readEmail("enron6/ham/"+line, hamDict, 'init', 0)
            elif 'spam' in line:
                spamNr+=1
                readEmail("enron6/spam/"+line, spamDict, 'init', 0)

        hamAllWordsNr = np.sum(hamDict.values())
        spamAllWordsNr = np.sum(spamDict.values())

        with open("stopwords.txt", "r") as f:
            text = f.read()
            text = text.split()
            for word in text:
                stopWords.append(word)

        with open("stopwords2.txt", "r") as f:
            text = f.read()
            text = text.split()
            for word in text:
                stopWords.append(word)

        added = False 
        for i in fileNr:
            wordsNrDict.clear()
            pHamDict.clear()
            pSpamDict.clear() 

            readEmail("ssl/"+str(i)+".txt", wordsNrDict, 'additiv', 1)

            # classification
            sum = 0
            for word in wordsNrDict:
                sum += wordsNrDict[word] * (math.log(pSpamDict[word]) - math.log(pHamDict[word]))
            # lnR > 0 --> spam      lnR < 0 --> ham
            lnR = math.log(spamNr/(spamNr+hamNr)) - math.log(hamNr/(spamNr+hamNr)) + sum

            # prediction accuracy + insert
            if lnR >= math.log(5):
                readEmail("ssl/"+str(i)+".txt", spamDict, 'init', 0)
                f = open('train.txt', 'a')
                f.write('\n' + "ssl/" + str(i) + ".txt")
                f.close()
                added = True
                fileNr.remove(i)
            elif lnR <= math.log(1/5):
                readEmail("ssl/"+str(i)+".txt", hamDict, 'init', 0)
                f = open('train.txt', 'a')
                f.write('\n' + "ssl/" + str(i) + ".txt")
                f.close()
                added = True
                fileNr.remove(i)


    print(fileNr)
    naivBaies("train.txt", 'TRAIN', 'additiv', 1)
        


def main():
    naivBaies("train.txt", 'TRAIN', 'train', 0)
    naivBaies("test.txt", 'TEST', 'train', 0)
    naivBaies("train.txt", 'TRAIN', 'additiv', 1)
    naivBaies("test.txt", 'TEST', 'additiv', 1)
    naivBaies("train.txt", 'TRAIN', 'additiv', 0.1)
    naivBaies("test.txt", 'TEST', 'additiv', 0.1)
    naivBaies("train.txt", 'TRAIN', 'additiv', 0.01)
    naivBaies("test.txt", 'TEST', 'additiv', 0.01)

    crossValidation("train.txt", 5)
    HalfSupervised()


if __name__=="__main__":
    main()