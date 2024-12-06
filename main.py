import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def get_clean_data():
    data = pd.read_csv('data.csv')
    data.drop(columns=['Unnamed: 32', 'id'], axis=1, inplace=True)
    data.diagnosis = [1 if value == 'M' else 0 for value in data.diagnosis]
    data['diagnosis'] = data['diagnosis'].astype('category', copy=False)

    return data

def create_model(data):
    y = data['diagnosis']  # target
    X = data.drop('diagnosis', axis=1)

    #scale the data
    scaler = StandardScaler()
    X=scaler.fit_transform(X)

    #split up the data
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

    #train
    model=LogisticRegression()
    model.fit(X_train, y_train)

    #model testing
    y_pred=model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, scaler



def main():
    data=get_clean_data()

    model, scaler = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
