import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

data = pd.read_csv('preprocessed.csv')

encode_venue = LabelEncoder()
encode_team = LabelEncoder()

data['venue'] = encode_venue.fit_transform(data['venue'])
data['batting_team'] = encode_team.fit_transform(data['batting_team'])
data['bowling_team'] = encode_team.fit_transform(data['bowling_team'])
data['toss_winner'] = encode_team.fit_transform(data['toss_winner'])
X = data.drop(['Pplay1_runs', 'pplay twick1',
       'Pplay2_runs', 'pplay twick2', 'toss_winner', 'toss_decision',
       'Pitch Type'], axis=1)
y = data['Pplay1_runs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

linear_regression = LinearRegression()
model = linear_regression.fit(X_train, y_train)

score = model.score(X_train, y_train)
print('train accuracy : ', score)

y_pred = model.predict(X_test)
test_score = metrics.r2_score(y_test, y_pred)
print("test accuracy : ", test_score)



joblib.dump(linear_regression, 'regression_model.joblib')
joblib.dump(encode_venue, 'venue_encoder.joblib')
joblib.dump(encode_team, 'team_encoder.joblib')