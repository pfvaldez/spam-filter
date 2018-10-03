from __future__ import division

from string import punctuation
from collections import Counter,OrderedDict
'''
NOTE: Return testing_folders --> email_folders[71:]
			 training_folders --> email_folders[:71]
'''
import math
import os
import re
import numpy as np
np.set_printoptions(threshold=np.nan)

'''
def function1(input):
	#do stuff here
	return output
'''
def parse_labels(label_dir):
	email_label = OrderedDict()
	with open(label_dir,'r') as label_file:
		for line_n,line in enumerate(label_file):
			line = line.strip()
			line = re.sub('[.]','',line).split(" ")
			#print("\nline:",line)
			email_label["trec06p-cs280"+line[1]] = line[0]
	# print(email_label)
	label_file.close()

	return(email_label)

def parse_words(email_label,ham_words,spam_words,all_words,n_ham,n_spam,folders,stopwords):
	'''
	parses words from email and generates vector of unique words
	'''

	#----------------------------------------------------------------------
	#HAM
	ham_word = []
	spam_word = []

	for folder in folders: 
		emails= [os.path.join(folder,f) for f in sorted(os.listdir(folder))]  
	

		for email in emails:
			label = email_label[email]
			words =()
			#print(label)
			if label == 'ham':
				n_ham += 1
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
									ham_word.append(word.lower())
				#_set = set(wordVec_training_ham)
				#for i in set(wordVec_training_ham):
				#	all_words_ham.append(i)

				#train_ham_set = list(_set)
				ham_words.append(ham_word)
				#print("train_ham",train_ham)

			else:
				n_spam += 1
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
									spam_word.append(word.lower()) 
				spam_words.append(spam_word)

			file.close()

	return (n_ham,n_spam,ham_words,spam_words,all_words)

def generate_dict(all_words):
	mc = Counter(all_words)
	mos=mc.most_common(10000)
	dictionary = OrderedDict(mos)
	print("len(All words):",len(all_words))
	print('len(dictionary)',len(dictionary))

	return (dictionary)

def generate_feat_mat(dictionary,words):


def main():
	'''
	1.Parse labels 
	2.Parse words from training-email
	3. generate dictionary
	3.Parse words from test-email
	4.Construct Feature Matrix - Training
	5.Construct Feature Matrix - Testing

	variables list:

	n_ham_train - '[[],[],...[]]'-used for prior_ham,-no. of ham emails in training folder
	n_spam_train- '[[],[],...[]]'-used for prior_spam,-no. of spam emails in training folder
	ham_words_train - '[[word1,..wordn],[word1,..wordn],...[word1,..wordn]]' - used for matrices_ham_T, -contains parsed words from ham emails in training folder
	spam_words_train - '[[word1,..wordn],[word1,..wordn],...[word1,..wordn]]' - used for matrices_spam_T, -contains parsed words from spam emails in training folder
	dictionary - ordered dictionary - used to generate feature matrices
	words_test- list -used for generating test matrices
	
	'''
	#----------------------------------------------------
	#global variables
	wordVec_training_ham = []
	wordVec_training_spam = []
	train_dir = "trec06p-cs280/data"
	label_dir = "trec06p-cs280/labels"

	email_folders = [os.path.join(train_dir,f) for f in sorted(os.listdir(train_dir))]
	training_folders = email_folders[:2]
	testing_folders = email_folders[71:73]
	# training_folders = email_folders[:71]
	# testing_folders = email_folders[71:]


	print("training folder range:",training_folders[0],"-",training_folders[-1])
	print("testing folder range:",testing_folders[0],"-",testing_folders[-1],"\n")

	n_spam_train = 0
	n_ham_train = 0
	n_spam_test = 0
	n_ham_test = 0

	train_spam = []
	train_ham = []
	test_spam = []
	test_ham = []
	ham_words_train = []
	spam_words_train = []
	ham_words_test = []
	spam_words_test = []
	all_words = []
	labels_test = []


	stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
	#----------------------------------------------------
	print('parsing labels...')
	email_label = parse_labels(label_dir)
	#print(email_label)

	print('parsing words from training folders...')
	(n_ham_train,n_spam_train,ham_words_train,spam_words_train,all_words)=parse_words(email_label,ham_words_train,spam_words_train,all_words,n_ham_train,n_spam_train,training_folders,stopwords)

	print("generating dictionary...")
	dictionary = generate_dict(all_words)

	print('parsing words from testing folders...')
	(n_ham_test,n_spam_test,ham_words_test,spam_words_test,all_words)=parse_words(email_label,ham_words_test,spam_words_test,all_words,n_ham_test,n_spam_test,testing_folders,stopwords)
	words_test = ham_words_test+spam_words_test

	print('\n\n creating feature matrices...')
	print('\n\n creating ham feature matrices...')


	return None

if __name__ == '__main__':
	main()