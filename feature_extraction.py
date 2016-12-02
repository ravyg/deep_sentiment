import nltk
import re
import string
import operator
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

file1 ='./auto_annotated.csv'
output_file = './Output/output.csv'


##########################################################################################
## 0 no, dont read word lists from file. Creates word list files
## 1 yes, read words lists from file. Does not create word list files

reading_words_from_file = 0
#########################################################################################

stemmer = PorterStemmer()
freq_yes_word_read_fromfile = []
freq_no_word_read_fromfile = []
stop_words_list = stopwords.words('english')
dictionary_freq_yes = {}
dictionary_freq_no = {}
f_output = open(output_file, 'w')   
words_dictionary_freq_no_sorted = []
words_dictionary_freq_yes_sorted = []
list_automatic_freq_yes = []
list_automatic_freq_no = []
file_Tweets = open(file1,'r')

#########################################################################################

if reading_words_from_file == 1:

    file_auto_yes_list = open('../word_lists/auto_yes_list_file.txt','r')
    file_auto_no_list = open('../word_lists/auto_no_list_file.txt','r')
   


    for word in file_auto_yes_list.readlines():
        word = word.replace('\n','')
        freq_yes_word_read_fromfile.append(word)

    for word in file_auto_no_list.readlines():
        word = word.replace('\n','')
       freq_no_word_read_fromfile.append(word)

 
          
    file_auto_yes_list.close() 
    file_auto_no_list.close()


if read_word_lists_from_file == 1:
    list_automatic_freq_no = freq_no_word_read_fromfile[:]
    list_automatic_freq_yes = freq_yes_word_read_fromfile[:]
  



##################################################################
if read_word_lists_from_file == 0:
    create_files_of_word_lists()




# sorting frequent no terms
print "dictionary_freq_no"
dictionary_freq_no_sorted = sorted(dictionary_freq_no.items(), key=operator.itemgetter(1) , reverse=True)
for tup in dictionary_freq_no_sorted:
    words_dictionary_freq_no_sorted.append(tup[0])

# sorting frequent Yes terms
print "   "
print "dictionary_freq_yes"
dictionary_freq_yes_sorted = sorted(dictionary_freq_yes.items(), key=operator.itemgetter(1) , reverse=True)
#print dictionary_freq_yes_sorted #[0:400]


# adding sorted frequent yes terms into the final word list
for tup in dictionary_freq_yes_sorted:
    words_dictionary_freq_yes_sorted.append(tup[0])



#Iterating on sorted No frequent terms and if NoFreqWord is not available in the YesFreqTerm then adding into final automatic list of No terms.

for word in words_dictionary_freq_no_sorted[0:11000]:
    if word not in words_dictionary_freq_yes_sorted:
        list_automatic_freq_no.append(word.lower())



#Iterating on sorted Yes frequent terms and if YesFreqWord is not available in the NoFreqTerm then adding into final automatic list of Yes terms.

for word in words_dictionary_freq_yes_sorted[0:11000]:
    if word not in words_dictionary_freq_no_sorted:
        list_automatic_freq_yes.append(word.lower())


for line in file_Tweets.readlines():
    the_vector = {}
    column = line.split(",")
    tweet_text = column[1]
    tweet_text = re.sub(r'[^\x00-\x7f]',r' ',tweet_text)
    tweet_text_tokens = word_tokenize(tweet_text)
    token_text_removed_stopwords = [i for i in tweet_text_tokens if i not in stop_words_list]
    if column[2] in ["Yes"]:
        for word in token_text_removed_stopwords: 
            word = word.lower()
            if word in dictionary_freq_yes:
                dictionary_freq_yes[word] = dictionary_freq_yes[word] + 1
            else:
                dictionary_freq_yes[word] = 1

    if column[0] in ["No"]:
        for word in token_text_removed_stopwords: 
            word = word.lower()
            if word in dictionary_freq_no:
                dictionary_freq_no[word] = dictionary_freq_no[word] + 1
            else:
                dictionary_freq_no[word] = 1
    tweet_pos_tag = nltk.pos_tag(token_text_removed_stopwords)
    
    the_vector['class'] = column[3]
    the_vector['tweet_id'] = column[0]
    the_vector['Adjective_count'] = func_tags_JJ(tweet_pos_tag)
    the_vector['Adverb_count'] = func_tags_RB(tweet_pos_tag)
    the_vector['Verb_count'] = func_VB(tweet_pos_tag)
    the_vector['ProperNoun_count'] = func_tags_NNP(tweet_pos_tag)
    the_vector['CapitalWordEx_count'] = func_CWEx(token_text_removed_stopwords)
    the_vector['auto_No_freq_terms'] = automatic_freq_no(text)
    the_vector['auto_Yes_freq_terms'] = automatic_freq_yes(text)
     print '###################################################'
    ## the_vector['id_val']
    f_output.write(the_vector['class'])
    f_output.write(',')
    f_output.write(str(the_vector['tweet_id']))
    f_output.write(',')
    f_output.write(str(the_vector['Adjective_count']))
    f_output.write(',')
    f_output.write(str(the_vector['Adverb_count']))
    f_output.write(',')
    f_output.write(str(the_vector['Verb_count']))
    f_output.write(',')
    f_output.write(str(the_vector['ProperNoun_count']))
    f_output.write(',')
    f_output.write(str(the_vector['CapitalWordEx_count']))
    f_output.write(',')
    f_output.write(str(the_vector['auto_No_freq_terms']))
    f_output.write(',')
    f_output.write(str(the_vector['auto_Yes_freq_terms']))
    f_output.write("\n")




   
f_output.write('tweet_id,Adjective_count,Verb_count,ProperNoun_count,CapitalWordEx_count,auto_No_freq_terms,auto_Yes_freq_terms')
f_output.write("\n")

file_Tweets.close()
#############Functions####################################################

def create_files_of_word_lists():

    file_auto_yes_list = open('../word_lists/auto_yes_list_file.txt','w')
    file_auto_no_list = open('../word_lists/auto_no_list_file.txt','w')
 
   
    for item in list_automatic_freq_no:
        file_auto_no_list.write(item)
        file_auto_no_list.write('\n')


    for item in list_automatic_freq_yes:
        file_auto_yes_list.write(item)
        file_auto_yes_list.write('\n')
        
    file_auto_no_list.close() 
    file_auto_yes_list.close()

def func_tags_NNP(tag_sequence):
    count_nnp = 0
    for tup in tag_sequence:    
        if tup[1] == 'NNP':
            count_nnp = count_nnp + 1
    return count_nnp


def func_tags_JJ(tag_sequence):
    count_adj = 0
    for tup in tag_sequence:
        if tup[1] == 'JJ' or tup[1] == 'JJR' or tup[1] == 'JJS':
            count_adj = count_adj + 1
    return count_adj

def func_tags_RB(tag_sequence):
    count_adverb = 0
    for tup in tag_sequence:
        if tup[1] == 'RB' or tup[1] == 'RBR' or tup[1] == 'RBS':
            count_adverb = count_adverb + 1
    return count_adverb

def func_VB(tag_sequence):
    count_verb = 0
    for tup in tag_sequence:
        if tup[1] == 'VB':
            count_verb = count_verb + 1
    return count_verb

def func_CWEx(tag_sequence):
    count_CWEx = 0
    for word in tag_sequence:
        if word.isupper():
            count_CWEx = count_CWEx + 1
    return count_CWEx


def automatic_freq_no(text):
    count_words = 0
    
    for word in text:
        word1 = stemmer.stem(word.lower())
        if word1 in list_automatic_freq_no:
            count_words = count_words + 1
    return count_words

def automatic_freq_yes(text):
    count_words = 0
    
    for word in text:
        word1 = stemmer.stem(word.lower())
        if word1 in list_automatic_freq_yes:
            count_words = count_words + 1
    return count_words



f_output.close()


print '<<<<<<<<<<<<<DONE>>>>>>>>>>>'
