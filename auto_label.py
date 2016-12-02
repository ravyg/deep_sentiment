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

reload(sys)
sys.setdefaultencoding('utf-8')
#
p.set_options(p.OPT.URL)
input_path = 'sample/'
# load the pickled emoji dictionary
emoji_pos = ed.emoji_list_pos
emoji_neg = ed.emoji_list_neg
annotated_tweet = []

for input_filename in os.listdir(input_path):
  # Ignore system files.
  dirpath = os.path.splitext(os.path.basename(input_filename))[0]
  if not input_filename.startswith('.'):
    for line in open(input_path + input_filename,'r').readlines():
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
            annotated_tweet.append(str(tweet['id_str']) + "," + cleaned_text + ',' + "Yes")
          # Negative.  
          elif any(emoji in raw_tweet_text for emoji in emoji_neg):
            annotated_tweet.append(str(tweet['id_str']) + "," + cleaned_text + ',' + "No")
            # break
          else:
            continue
      # Skip bad tweet.        
      except:
        continue

print annotated_tweet
np.savetxt('auto_annotated.csv', annotated_tweet, delimiter=",", fmt="%s")


