import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)  # Inicializa pesos aleatórios
        self.bias = np.random.randn()  # Inicializa o bias aleatório
        self.learning_rate = 0.1  # Taxa de aprendizado

    def activation(self, x):
        return 1 if x >= 0 else 0  # Função de ativação degrau

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                # Atualização dos pesos (Regra Delta)
                self.weights += self.learning_rate * error * np.array(inputs)
                self.bias += self.learning_rate * error
                total_loss += abs(error)

            print(f"Epoch {epoch + 1}/{epochs} - Erro: {total_loss}")


# Dados de treinamento (AND e OR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # Saída para lógica AND
y_or = np.array([0, 1, 1, 1])  # Saída para lógica OR

# Criando e treinando a IA
print("Treinando para a função lógica AND...")
nn_and = SimpleNeuralNetwork(input_size=2)
nn_and.train(X, y_and)

print("\nTreinando para a função lógica OR...")
nn_or = SimpleNeuralNetwork(input_size=2)
nn_or.train(X, y_or)

# Testando a IA
print("\nTestando a IA para lógica AND:")
for i, test in enumerate(X):
    print(f"Entrada: {test} -> Saída: {nn_and.predict(test)}")

print("\nTestando a IA para lógica OR:")
for i, test in enumerate(X):
    print(f"Entrada: {test} -> Saída: {nn_or.predict(test)}")
