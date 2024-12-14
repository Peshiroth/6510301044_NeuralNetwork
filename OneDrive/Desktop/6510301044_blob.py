# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. สร้างชุดข้อมูล 2D ด้วย make_blobs
X, y = make_blobs(n_samples=200, centers=[[2, 2], [3, 3]], cluster_std=0.75, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. แสดงข้อมูลก่อนการ train
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor='k')
plt.title("Generated Data")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.show()

# 3. สร้าง Neural Network
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),  # Hidden Layer 1
    Dense(4, activation='relu'),                   # Hidden Layer 2
    Dense(1, activation='sigmoid')                 # Output Layer
])

# 4. Compile และ Train โมเดล
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, verbose=0)

# 5. Plot Decision Boundary
def plot_decision_boundary(X, y, model):
    # สร้าง grid สำหรับ plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict บน grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).reshape(xx.shape)
    
    # Plot Decision Boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.show()

plot_decision_boundary(X, y, model)
