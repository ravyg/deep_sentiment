#!/usr/bin/env python
# This file does the part to automated annotation.
# -*- coding: utf-8 -*-
import json
import os
import sys
import re
import numpy as np
import preprocessor as p
import emoji_dict as ed
from pycorenlp import StanfordCoreNLP
import subprocess
# Reload System.
reload(sys)
sys.setdefaultencoding('utf-8')
# Building Tweet Pre-Processor Object.
p.set_options(p.OPT.URL)
input_path = 'data/'
root = os.getcwd()

# Get Emojis Listing.
emoji_pos = ed.emoji_list_pos
emoji_neg = ed.emoji_list_neg
# Run this for each day collected tweets.
for input_filename in os.listdir(input_path):
  # Ignore system files.
  # dirpath = os.path.splitext(os.path.basename(input_filename))[0]
  if not input_filename.startswith('.'):
    annotated_tweet = []
    tweets_for_corenlp = []
    tweets_id_order = []
    tweet_count = 0
    for line in open(root + '/' + input_path + input_filename,'r').readlines():
      current_filename = os.path.splitext(os.path.basename(input_filename))[0]
      # Pre validate json file lines move to next if bad json.
      try:
        tweet = json.loads(line)
        # Check for not a re-tweet and is in English.
        if tweet.get('lang') == 'en' and 'text' in tweet.keys() and not tweet['text'].startswith('RT'):
          # print(json.dumps(tweet, indent=2))
          # Create file name to store.
          filename = tweet.get('id_str')
          # Keyword searcher.
          raw_tweet_text = tweet.get('text')
          # Clean raw text
          cleaned_text = p.clean(raw_tweet_text)
          # Remove comma as we store in csv it confuse later system.
          cleaned_text = cleaned_text.replace(',', ' ')
          if any(emoji in raw_tweet_text for emoji in emoji_pos):
            annotated_tweet.append(str(tweet['id_str']) + "," + cleaned_text + ',' + "Positive")
          # Negative.  
          elif any(emoji in raw_tweet_text for emoji in emoji_neg):
            annotated_tweet.append(str(tweet['id_str']) + "," + cleaned_text + ',' + "Negative")
          else:
            # build cleaned tweets set.
            super_cleaned_text = re.sub('[^a-zA-Z]+', ' ', cleaned_text)
            #super_cleaned_text = [i for i in super_cleaned_text if len(i) > 2]
            shortword = re.compile(r'\W*\b\w{1,3}\b')
            super_cleaned_text = shortword.sub('', super_cleaned_text)
            super_cleaned_text = re.sub('[ ]+', ' ', super_cleaned_text)
            tweets_for_corenlp.append(super_cleaned_text)
            tweets_id_order.append(str(tweet['id_str']) + "," + cleaned_text)
      # Skip bad tweet.        
      except:
        continue
      tweet_count = tweet_count + 1
      print "PROCESSING TWEET: " + str(tweet_count)
    # create all tweets input file for sentiment via corenlp.
    with open('temp_tweet_text.txt', 'w') as outfile:
      outfile.write('.\n\n'.join(tweets_for_corenlp) + '\n')
    # Feed temp tweet text to corenlp.
    os.chdir('stanford-corenlp-full-2016-10-31')
    stanfordcommandSenti = 'java -cp "*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file ../temp_tweet_text.txt > ../classification.txt'
    os.system(stanfordcommandSenti)
    # Read output file created by corenlp.
    with open('../classification.txt', 'r') as corenlf_classification:
      count = 0
      variable_sent = []
      for key, classification in enumerate(corenlf_classification):
          count+=1
          if count % 2 == 0: 
            variable_sent.append(re.sub('[^a-zA-Z]+', '', classification))
      inputs = dict(zip(tweets_id_order, variable_sent))
      for key, value in inputs.iteritems():
        if value is not 'Neutral':
          annotated_tweet.append(key + ',' + value)
    os.chdir('../')
    # Save results per file.
    np.savetxt('auto_annotated/auto_annotated_' + current_filename + '.csv', annotated_tweet, delimiter=",", fmt="%s")
    print annotated_tweet

