import json
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
DATA_URL = r"https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/events/15978.json"

try:
    response = requests.get(DATA_URL)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from GitHub: {e}")
    exit()

shots = [x for x in data if x['type']['name'] == 'Shot']
df = pd.json_normalize(shots)
print(f"Loaded {len(df)} shots")

df = df[[
    'team.name',
    'player.name',
    'location',
    'shot.end_location',
    'shot.body_part.name',
    'shot.technique.name',
    'shot.outcome.name',
    'shot.type.name',
    'shot.first_time',
    'shot.one_on_one',
    'under_pressure'
]]

def calculate_distance(x, y):
    goal_x, goal_y = 120, 40
    return np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)


def calculate_angle(x, y):
    goal_x, goal_y1, goal_y2 = 120, 36, 44
    a = np.sqrt((goal_x - x) ** 2 + (goal_y1 - y) ** 2)
    b = np.sqrt((goal_x - x) ** 2 + (goal_y2 - y) ** 2)
    c = 8
    cos_value = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    cos_value = np.clip(cos_value, -1, 1)
    return np.arccos(cos_value)

df['x'] = df['location'].apply(lambda loc: loc[0])
df['y'] = df['location'].apply(lambda loc: loc[1])
df['distance'] = df.apply(lambda r: calculate_distance(r.x, r.y), axis=1)
df['angle'] = df.apply(lambda r: calculate_angle(r.x, r.y), axis=1)

df['is_goal'] = df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)

X = df[['distance', 'angle']]
y = df['is_goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"Model trained | AUC = {auc:.3f}")
print("\nxG Prediction")
print("Enter shot coordinates (x, y) or type 'exit' to quit.")

while True:
    user_input = input("\nEnter x and y (example: 100 35): ").strip()
    if user_input.lower() == 'exit':
        print("NOoooo")
        break

    try:
        x, y = map(float, user_input.split())
        dist = calculate_distance(x, y)
        ang = calculate_angle(x, y)

        new_shot_data = pd.DataFrame([[dist, ang]], columns=['distance', 'angle'])

        xg = model.predict_proba(new_shot_data)[0][1]

        print(f"Expected Goals (xG) = {xg:.3f}")
    except Exception:
        print("Invalid input")
