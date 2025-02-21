import numpy as np

class SmartDecisionAI:
    def __init__(self):
        self.weights = np.random.rand(3)  # Pesos para cada tipo de opção
        self.bias = np.random.rand()
        self.learning_rate = 0.1

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def predict(self, options):
        values = np.array([len(opt) for opt in options])  # Usa tamanho das frases como base
        probabilities = self.softmax(values * self.weights + self.bias)
        choice_index = np.argmax(probabilities)
        return options[choice_index], probabilities

    def train(self, options, best_choice_index):
        values = np.array([len(opt) for opt in options])
        probabilities = self.softmax(values * self.weights + self.bias)

        # Atualiza pesos usando erro
        error = np.zeros(3)
        error[best_choice_index] = 1 - probabilities[best_choice_index]

        self.weights += self.learning_rate * error * values
        self.bias += self.learning_rate * error.sum()

        return probabilities

# Criando IA
ai = SmartDecisionAI()

while True:
    print("\nDigite três opções (simples, mediana e complexa):")
    op1 = input("Opção simples: ")
    op2 = input("Opção mediana: ")
    op3 = input("Opção complexa: ")

    escolha, probs = ai.predict([op1, op2, op3])
    print(f"\n🔹 A IA escolheu: **{escolha}** (Confiança: {probs.max():.2f})")

    feedback = input("Essa escolha foi boa? (s/n): ").strip().lower()
    if feedback == "s":
        print("✅ Aprendizado confirmado!")
    else:
        best_index = int(input("Digite 0 para simples, 1 para mediana ou 2 para complexa (melhor opção): "))
        ai.train([op1, op2, op3], best_index)
        print("🔄 Aprendizado atualizado!")

    continuar = input("Deseja testar de novo? (s/n): ").strip().lower()
    if continuar != "s":
        break
