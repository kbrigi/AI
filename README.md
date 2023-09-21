# AI
Projects related to Artificial Intelligence: 2048 game solver, spam/ham email filtering 

# 2048 
It generates a random game starting position and plays it automatically to the end, generating game trees and following the most optimal path.
Following heuristics are applied when choosing the best move:
* the number of free cells
* the maximum of the numbers

The results can be seen in the out.txt file.

The winning accuracy is 75%.

# Email Spam Filtering Using Naive Bayes Classifier
**Naiv Bayes**

After calculating the probability that a word is spam/ham and building dictionaries with the help of the given files(described bellow),
the given email will be classified as ham/spam by analyzing each word and checking the probability of it being ham/spam.

**K-fold Cross Validation**

The dataset is divided into k subsets or folds. The model is trained and evaluated k times, using a different fold as the validation set each time. 
Performance metrics from each fold are averaged to estimate the model's generalization performance.

**Semi-supervised Learning of Naive Bayes**

After each email classification, it's data will be added to the trained dataset/accuracy prediction. This way new words will be labeled and the trained data gets bigger.

**Files**

ssl folder: 1000 txt emalis (input for classification)

enron6 folder: ham(1500) and spam(4500) already classified emails for learning 

test.txt: contains the name of the test files

train.txt: contains the name of the train files

stopwords.txt: for filtering common words, so they don't affect the classification
