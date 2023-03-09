label_lists = {'memotion7k': [4, 4, 4, 2, 3],
               'multioff':[2],
               'fbhm':[2]}

labels_multioff = {'Non-offensiv':0,
                   'offensive':1}

labels_memotion7k = {'overall_sentiment':
                     {'negative': 0,
                         'very_negative': 0,
                         'positive': 2,
                         'very_positive': 2,
                         'neutral': 1},
                    'humour':
                     {'not_funny': 0,
                      'funny': 1,
                      'very_funny': 2,
                      'hilarious': 3},
                     'sarcasm':
                     {'not_sarcastic': 0,
                         'general': 1,
                         'twisted_meaning': 2,
                         'very_twisted': 3},
                     'offensive':
                     {'not_offensive': 0,
                         'slight': 1,
                         'very_offensive': 2,
                         'hateful_offensive': 3, },
                     'motivational':
                     {'not_motivational': 0,
                         'motivational': 1
                      },

                     }
