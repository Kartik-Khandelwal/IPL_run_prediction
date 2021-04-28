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

   test_case['venue'] = venue_encoder.transform(test_case['venue'])
   test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
   test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])

   test_case = test_case[['venue', 'batting_team', 'bowling_team']]

   # test_array = test_case.to_numpy()

   # test_case = np.concatenate((np.eye(42)[test_array[:,0]],
   #                             np.eye(15)[test_array[:,1] -1],
   #                             np.eye(15)[test_array[:,2]]
   # ), axis = 1)
   

   return regressor.predict(test_case)