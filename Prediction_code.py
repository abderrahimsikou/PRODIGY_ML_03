import cv2
import joblib

# Upload model
model      = joblib.load('model.pkl')

# Test image
test_image = 'test_image/img1.jpg'
img        = cv2.imread(test_image, 0)
img        = cv2.resize(img, (64,64))
img        = img.flatten().reshape(1,-1)

# Prediction
prediction = model.predict(img)

if prediction[0] == 0:
    print('cat')
else:
    print('dog')
