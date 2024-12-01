import librosa
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
project_dir = Path(__file__).resolve().parents[2]
ravdess_dir = project_dir / "data" / "RAVDESS"

def extract_feature(file_name, mfcc = True, chroma = True, mel = True,contrast=True, tonnetz=True, zcr=True, centroid=True):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma or contrast or tonnetz or centroid:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y = X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y = X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    if zcr:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y = X).T, axis=0)
        result = np.hstack((result, zcr))
    if centroid:
        centroid = np.mean(librosa.feature.spectral_centroid(y = X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, centroid))
    return result

#Emotion in the RAVDESS dataset
emotions={
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

#Emotions to observe
observed_emotions = ['neural', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def load_and_balance_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob(str(ravdess_dir / '**' / '*.wav'), recursive=True):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions.get(emotion_code, None)
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True, zcr=True, centroid=True)
        x.append(feature)
        y.append(emotion)
    x, y = np.array(x), np.array(y)
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    x, y = smote.fit_resample(x, y)
    return train_test_split(x, y, test_size=test_size, random_state=9)

#split the dataset
x_train, x_test, y_train, y_test = load_and_balance_data(test_size=0.25)

#Standarlize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
param_grid = {
    'hidden_layer_sizes': [(512, 256), (256, 128)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'activation': ['relu', 'tanh']
}
grid = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)
#MLP Classifier
#Initialize the Multi Layer Perceptron Classifier
model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

#Predict for the test_set
y_pred = model.predict(x_test)

#Calculate the accuracy of model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))


from joblib import dump, load
# Lưu mô hình vào thư mục 'models'
model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "audio_emotion_model.pkl"
dump(model, model_path)
print(f"Mô hình đã được lưu tại: {model_path}")
