import pandas as pd
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))

    def sigmoid(self, x):
        clipped_x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-clipped_x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            activations.append(self.sigmoid(np.dot(activations[i], self.weights[i]) + self.biases[i]))
        return activations

    def backward(self, X, y, activations):
        error = y - activations[-1]
        deltas = [error * self.sigmoid_derivative(activations[-1])]
        
        for i in range(len(activations) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)

# อ่านข้อมูลจากไฟล์ Excel
df = pd.read_excel('train/Train Dataset1.xlsx')  # แทนชื่อไฟล์ที่เหมาะสมที่นี่

# แยกข้อมูลเป็นตัวแปรอิสระและตัวแปรตาม
X = df.iloc[:, 2:].values  # แถวที่ 3 เป็นตัวแปรอิสระ
y = df.iloc[:, 1].values.reshape(-1, 1)  # แถวที่ 2 เป็นตัวแปรตาม

# กำหนดพารามิเตอร์ของโมเดล
input_size = X.shape[1]
hidden_layers = [4, 4]  # กำหนดจำนวนโหนดในแต่ละชั้นซ่อน
output_size = 1  # หรือจำนวนคลาส
learning_rate = 0.01

# สร้างและฝึกโมเดล MLP
mlp = MLP(input_size, hidden_layers, output_size, learning_rate)
mlp.train(X, y, epochs=1000)  # ปรับจำนวนรอบการฝึกตามความเหมาะสม
