import numpy as np
import pandas as pd 
import joblib

def predictRuns(input_test):

   with open('regression_model.joblib', 'rb') as f:
      regressor = joblib.load(f)
   with open('venue_encoder.joblib', 'rb') as f:
      venue_encoder = joblib.load(f)
   with open('team_encoder.joblib', 'rb') as f:
      team_encoder = joblib.load(f)

   test_case = pd.read_csv(input_test)

   b = len(list(test_case['batsmen'])[0].split(","))

   if 'Narendra Modi Stadium' in list(test_case['venue']):
      if b == 2:
         a = 10
      elif b == 3:
         a = 5
      elif b == 4:
         a = -5
      else:
         a = -10
   elif 'Arun Jaitley Stadium' in list(test_case['venue']):
      if b == 2:
         a = 10
      elif b == 3:
         a = 5
      elif b == 4:
         a = -5
      else:
         a = -10
   elif 'Eden Garden' in list(test_case['venue']):
      if b == 2:
         a = 10
      elif b == 3:
         a = 5
      elif b == 4:
         a = -5
      else:
         a = -10
   elif 'M Chinnaswamy Stadium' in list(test_case['venue']):
      if b == 2:
         a = 10
      elif b == 3:
         a = 5
      elif b == 4:
         a = -5
      else:
         a = -10
   
   test_case['venue'] = venue_encoder.transform(['Feroz Shah Kotla'])
   test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
   test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])

   test_case = test_case[['venue', 'batting_team', 'bowling_team']]

   run = regressor.predict(test_case)
   
   return (round(run[0]) + a)