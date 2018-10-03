from __future__ import division

import math
import os
import re
import numpy as np
np.set_printoptions(threshold=np.nan)

from string import punctuation
from collections import Counter,OrderedDict

train_dir = "trec06p-cs280/data"
label_dir = "trec06p-cs280/labels"

email_folders = [os.path.join(train_dir,f) for f in sorted(os.listdir(train_dir))]

training_folders = email_folders[:2]
testing_folders = email_folders[72:74]
# training_folders = email_folders[:71]
# testing_folders = email_folders[71:127]


print("training folder range:",training_folders[0],"-",training_folders[-1])
print("testing folder range:",testing_folders[0],"-",testing_folders[-1])
print("parsing....")

n_spam_train = 0
n_ham_train = 0
n_spam_test = 0
n_ham_test = 0
_emails_train = 0

ham_words_train = []
spam_words_train = []
labels_test = []
trueL=[]
stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']

#--------------------------------------------------------------------------------------------
# STEP 1
# parsing labels
email_label = OrderedDict()
with open(label_dir,'r') as label_file:
	for line_n,line in enumerate(label_file):
		line = line.strip()
		line = re.sub('[.]','',line).split(" ")
		email_label["trec06p-cs280"+line[1]] = line[0]
label_file.close()

#--------------------------------------------------------------------------------------------
# STEP 2
# parsing words from training emails
print('during training....')
all_words= list()

for folder in training_folders: 
	emails= [os.path.join(folder,f) for f in sorted(os.listdir(folder))]  

	for email in emails:
		#print('email',email)
		_emails_train+=1
		label = email_label[email]
		ham_word = []
		spam_word = []
		words =()
		#print(label)
		if label == 'ham':
			n_ham_train += 1
			file=open(email,encoding='latin-1',mode='r')
			lines=file.readlines()
			for line in lines:
				line=line.strip()
				line=line.split(' ')
				for word in line:
					word=word.rstrip('.')
					word=word.rstrip(',')
					if word.isalpha():
						if re.match('[^a-z]',word.lower()):
							pass
						else:
							if word.lower() not in stopwords:
								all_words.append(word.lower())
								ham_word.append(word.lower())
			ham_words_train.append(ham_word)


		else:
			n_spam_train += 1
			file=open(email,encoding='latin-1',mode='r')
			#print('file.read():\n',file.read())
			lines=file.readlines()
			for line in lines:
				line=line.strip()
				line=line.split(' ')
				for word in line:
					word=word.rstrip('.')
					word=word.rstrip(',')
					if word.isalpha():
						if re.match('[^a-z]',word.lower()):
							pass
						else:
							if word.lower() not in stopwords:
								all_words.append(word.lower())
								spam_word.append(word.lower()) 
			spam_words_train.append(spam_word)
		file.close()

prior_ham = n_ham_train/(n_ham_train+n_spam_train)
prior_spam = n_spam_train/(n_ham_train+n_spam_train)

print("len(All words):",len(all_words))
mc = Counter(all_words)
mos=mc.most_common(10000)
#print("most_common10k:", mos)
dictionary = OrderedDict(mos)
print('len(dictionary)',len(dictionary))
print('n_ham_train + n_spam_train:',n_ham_train+n_spam_train)
# print('emails',_emails_train)
# print('prior_ham:',prior_ham)
# print('prior_spam:',prior_spam)

#--------------------------------------------------------------------------------------------
# STEP 3
# parsing words from testing emails
print('during testing....')
#filesTemp=[]

words_test = []
ham_word_test = []
spam_word_test = []
ham_words_test = []
spam_words_test = []

for folder in testing_folders:
	emails= [os.path.join(folder,f) for f in sorted(os.listdir(folder))]  
	for email in emails:
		label = email_label[email]
		words = []
		if label == 'ham':
			labels_test.append(label)
			n_ham_test += 1
			file=open(email,encoding='latin-1',mode='r')
			for line in lines:
				line=line.strip()
				line=line.split(' ')
				for word in line:
					word=word.rstrip('.')
					word=word.rstrip(',')
					if word.isalpha():
						if re.match('[^a-z]',word.lower()):
							pass
						else:
							if word.lower() not in stopwords:
								ham_word_test.append(word.lower())
			words_test.append(ham_word_test)


		elif label == 'spam':
			n_spam_test += 1
			file=open(email,encoding='latin-1',mode='r')
			lines=file.readlines()
			for line in lines:
				line=line.strip()
				line=line.split(' ')
				for word in line:
					word=word.rstrip('.')
					word=word.rstrip(',')
					if word.isalpha():
						if re.match('[^a-z]',word.lower()):
							pass
						else:
							if word.lower() not in stopwords:
								spam_word_test.append(word.lower())
			words_test.append(spam_word_test)
		file.close()

#--------------------------------------------------------------------------------------------
#STEP 4
#generate feature matrix from training emails

print("\n\ngenerating feature matrices...")

# ham feature matrix
print("generating training ham feature matrices...")
keys=list(dictionary.keys())
#print(keys)
matrices_ham_T = []
for words in ham_words_train:
	row = [0]*len(dictionary.keys())
	for word in words:
		if word in keys:
			index = keys.index(word)
			row[index] = 1

	matrices_ham_T.append(row)
npmatrices_ham_T=np.asarray(matrices_ham_T)
npmatrices_ham=np.transpose(npmatrices_ham_T)
#print("npmatrices_ham:\n",npmatrices_ham)
(rows_ham_T,columns_ham_T) = npmatrices_ham_T.shape
print('rows_ham_T,columns_ham_T:',rows_ham_T,columns_ham_T)

# spam feature matrix
print("generating training spam feature matrices...")
matrices_spam_T = []
for words in spam_words_train:
	matrix_spam_T = [0]*len(dictionary.keys())
	for word in words:
		if word in keys:
			index=keys.index(word)
			matrix_spam_T[index]=1

	matrices_spam_T.append(matrix_spam_T)
npmatrices_spam_T=np.asarray(matrices_spam_T)
npmatrices_spam=np.transpose(npmatrices_spam_T)
(rows_spam_T,columns_spam_T) = npmatrices_spam_T.shape
print('rows_spam_T,columns_spam_T:',rows_spam_T,columns_spam_T)

#--------------------------------------------------------------------------------------------
# STEP 5
# generate feature matrix from testing emails

print("generating test feature matrices...")
matrices_test_T = []
n_words = 0
print("len(words_test):",len(words_test))
for words in words_test:
	n_words+=1
	print("iteration of words: %d of %d" %(n_words,len(words_test)))
	matrix_test_T = [0]*len(dictionary.keys())
	for word in words:
		if word in keys:
			#print(word)
			index=keys.index(word)
			matrix_test_T[index]=1

	matrices_test_T.append(matrix_test_T)
print('rows_test_T,columns_test_T:',rows_spam_T,columns_spam_T)

#--------------------------------------------------------------------------------------------
# STEP 6
# computing individual probabilities + lambda smoothing

_lambda = 1
ham_likelihood_array = ((np.sum(npmatrices_ham,axis=1)+_lambda))/(n_ham_train +(len(dictionary)*_lambda))
ham_likelihood = np.sum(ham_likelihood_array)

spam_likelihood_array = (np.sum(npmatrices_spam,axis=1)+_lambda)/(n_spam_train+(len(dictionary)*_lambda))
spam_likelihood = np.sum(spam_likelihood_array)

# --------------------------------------------------------------------------------------------
# STEP 7
# classifyin emails as ham or spam
print("\nclassifying test emails...")
_list = list()

test_classify= []
#print(len(matrices_test_T[0]),len(matrices_test_T[-1]))
#for document in matrices_test_T:
for folder in testing_folders: 
	emails= [os.path.join(folder,f) for f in sorted(os.listdir(folder))]  
	for email in emails:
		print("email:",email)
		wordVec = set()
		filesTemp.append(email)
		label = email_label[email]
		words = ()
		trueL.append(label)

		similar = set(keys).intersection(wordVec)
		dVector=[0]*len(keys)

		for word in similar:
			wordindex=keys.index(word)
			dVector[index]=1
		spam_probability=np.power(np.array(spam_likelihood_array),np.array(dVector))
		spam_probability_compliment=np.power(np.subtract(1,spam_likelihood_array),np.subtract(1,np.array(dVector)))

		spamp=np.sum(np.log(np.multiply(spam_probability,spam_probability_compliment)))+np.log(prior_spam)

		ham_probability=np.power(np.array(ham_likelihood_array),np.array(dVector))
		ham_probability_compliment=np.power(np.subtract(1,ham_likelihood_array),np.subtract(1,np.array(dVector)))

		hamp=np.sum(np.log(np.multiply(ham_probability,ham_probability_compliment)))+np.log(prior_ham)

		if hamp >= spamp:
			# print("ham")
			test_classify.append("ham")
		else:
			# print("spam")
			test_classify.append("spam")

f = open('ouput-test.txt','w')
f.write('filesTemp[i]+"\t\t"+labels_test[i]+"\t\t"+test_classify[i]+"\n"')
for i in range(len(labels_test)):
	#print(filesTemp[i],test_classify[i],labels_test[i])
	f.write(filesTemp[i]+"\t"+labels_test[i]+"\t"+test_classify[i]+"\n")
f.close()
#------------------------------------------------
TP = 0
TN = 0
FP = 0
FN = 0
precision = 0
print(len(test_classify))
#print(len(labels_test))
labels_test=trueL[21300:]
print(len(labels_test))
print(labels_test[0])
print(labels_test[-1])
for i in range(len(labels_test)):
	if labels_test[i] == "spam" and test_classify[i] == "spam":
		TP+=1
	elif labels_test[i] == "ham" and test_classify[i] == "ham":
		TN+=1
	elif labels_test[i] == "ham" and test_classify[i] == "spam":
		FP+=1
	elif labels_test[i] == "spam" and test_classify[i] == "ham":
		FN+=1

print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("TP+TN+FP+FN:",TP+TN+FP+FN)
print("precision",precision)
print("recall",recall)