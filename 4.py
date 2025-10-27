import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the data
digits = load_digits()
X = digits.data  # shape (1797, 64) - each image is 8x8 pixels
y = digits.target.reshape(-1, 1)  # shape (1797, 1)

# Normalize pixel values (0–16) to (0–1)
X = X / 16.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)  # ensure dense array
y_encoded = encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 2: Define the ANN class
class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m

    def backward(self, X, Y_true):
        m = Y_true.shape[0]
        dZ2 = self.A2 - Y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y, Y_pred)
            self.backward(X, Y)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=1)

# Step 3: Train the ANN model
ann = SimpleANN(input_size=64, hidden_size=32, output_size=10, learning_rate=0.5)
ann.train(X_train, y_train, epochs=1000)

# Step 4: Evaluate on test set
y_pred = ann.predict(X_test)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# Step 5: Visualize predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')

    true_label = y_true[i]
    pred_label = y_pred[i]

    # Green if correct, Red if wrong
    color = "green" if true_label == pred_label else "red"
    plt.title(f"T:{true_label} | P:{pred_label}", color=color)

    plt.axis('off')

plt.tight_layout()
plt.show()
