#!/usr/bin/env python
# This file does the part to automated annotation.
# -*- coding: utf-8 -*-
import json
import os
import sys
import re
import preprocessor as p
import emoji_dict as ed

reload(sys)
sys.setdefaultencoding('utf-8')

#p.set_options(p.OPT.URL, p.OPT.EMOJI)
p.set_options(p.OPT.URL)
input_path = 'data/'
# load the pickled emoji dictionary
emoji_dict = ed.emoji_dict

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
          tokens = p.tokenize(cleaned_text)
          for token in tokens:
            if token in emoji_dict.keys():
              print cleaned_text
          # @TODO: check if tweet have emoj.
      # Skip bad tweet.        
      except:
        continue