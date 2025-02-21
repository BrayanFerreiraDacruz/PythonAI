import numpy as np
import sqlite3

class SmartDecisionAI:
    def __init__(self, db_path="ai_memory.db"):
        self.db_path = db_path
        self.learning_rate = 0.1
        self.create_db()
        self.weights, self.bias = self.load_weights()

    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_weights (
                id INTEGER PRIMARY KEY,
                weight1 REAL,
                weight2 REAL,
                weight3 REAL,
                bias REAL
            )
        """)
        conn.commit()
        conn.close()

    def save_weights(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ai_weights")
        cursor.execute("INSERT INTO ai_weights (weight1, weight2, weight3, bias) VALUES (?, ?, ?, ?)",
                       (self.weights[0], self.weights[1], self.weights[2], self.bias))
        conn.commit()
        conn.close()

    def load_weights(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT weight1, weight2, weight3, bias FROM ai_weights LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            print("🔹 Pesos carregados do banco de dados!")
            return np.array([row[0], row[1], row[2]]), row[3]
        else:
            print("⚠ Nenhum peso encontrado. Iniciando aleatoriamente.")
            return np.random.rand(3), np.random.rand()

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def predict(self, options):
        values = np.array([len(opt) for opt in options])
        probabilities = self.softmax(values * self.weights + self.bias)
        choice_index = np.argmax(probabilities)
        return options[choice_index], probabilities

    def train(self, options, best_choice_index):
        values = np.array([len(opt) for opt in options])
        probabilities = self.softmax(values * self.weights + self.bias)

        error = np.zeros(3)
        error[best_choice_index] = 1 - probabilities[best_choice_index]

        self.weights += self.learning_rate * error * values
        self.bias += self.learning_rate * error.sum()

        self.save_weights()
        return probabilities

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
        print("🔄 Aprendizado atualizado e salvo!")

    continuar = input("Deseja testar de novo? (s/n): ").strip().lower()
    if continuar != "s":
        break
