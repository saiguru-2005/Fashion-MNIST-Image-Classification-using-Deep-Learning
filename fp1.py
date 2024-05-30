'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import layers, models

data=pd.read_csv("fashion-mnist_train.csv")
x=data.iloc[:,1].values
y=data.iloc[:,0].values
X=x.reshape(-1,28,28) / 255.0
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics= ['accuracy'])

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
test_loss,test_acc = model.evaluate(x_test,y_test)
print(f"Test accuracy:{test_acc}")
num_images= 5
plt.figure(figsize=(10,4))
for i in range(num_images):
    test_images=x_test[i].reshape(1,28,28,1)
    predicted_label=np.argmax(model.predict(test_images))
    
    plt.subplot(1,num_images,i+1)
    plt.imshow(x_test[i].reshape(28,28),cmp='gray')
    plt.title(f"Predicted : {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import layers, models

# Read the CSV file
data = pd.read_csv("fashion-mnist_train.csv")

# Extract features (pixel values) and labels
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Reshape and normalize the input features
x = x.reshape(-1, 28, 28, 1) / 255.0

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot some sample images and their predicted labels
num_images = 5
plt.figure(figsize=(10, 4))
for i in range(num_images):
    test_image = x_test[i]
    predicted_label = np.argmax(model.predict(test_image.reshape(1, 28, 28, 1)))
    
    plt.subplot(1, num_images, i + 1)
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
