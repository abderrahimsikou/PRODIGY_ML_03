import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

dir = 'C:\\Users\\FUJI\\Desktop\\Cats and Dogs classification (SVM)\\dataset'

categories = ['cat','dog']

x = []
y = []

for category in categories:
    label = 0 if category == "cat" else 1
    
    path = os.path.join(dir,category)
    
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        
        image = cv2.imread(img_path,0)
        image = cv2.resize(image, (64,64)) # Resize image 
        
        x.append(image.flatten())
        y.append(label)
        
x = np.array(x)
y = np.array(y)

# Train Test Split Data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Train model
svm_model  = SVC(kernel='rbf',C=1,gamma='scale')
svm_model.fit(x_train,y_train)

# Accuarcy account
prediction = svm_model.predict(x_test)     
accuracy   = accuracy_score(prediction,y_test)
print('accuarcy:', accuracy * 100, '%') 

# Save model
joblib.dump(svm_model, 'model.pkl')                       
