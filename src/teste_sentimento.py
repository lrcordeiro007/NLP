import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Certifique-se de que os dados do NLTK estão baixados
nltk.download('stopwords')

# 1. Criar um conjunto de dados de exemplo
texts = []
labels = []
with open('dados.txt', 'r', encoding='utf-8') as f:
    for linha in f:
        # Espera-se que cada linha seja: texto[TAB]label
        partes = linha.strip().rsplit('\t', 1)
        if len(partes) == 2:
            texts.append(partes[0])
            labels.append(partes[1])

# Se não houver dados, use os exemplos antigos
if not texts:
    texts = [
        "A empresa tem um ótimo atendimento ao cliente.",
        "O produto é de baixa qualidade e não funciona.",
        "Excelente serviço, superou minhas expectativas!",
        "Péssimo suporte, demoraram para me responder.",
        "Estou muito satisfeito com o resultado final.",
        "Que decepção! Não recomendo de jeito nenhum.",
        "Melhor experiência de compra que já tive.",
        "Estou decepcionado com a entrega atrasada.",
        "Pior produto que já comprei na vida."
    ]
    labels = ['positivo', 'negativo', 'positivo', 'negativo', 
              'positivo', 'negativo', 'positivo', 'negativo', 'negativo']

# 2. Pré-processamento e Vetorização
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=stopwords.words('portuguese')
)

# 3. Dividir os dados para treinamento e teste
# Adicione 'stratify=labels' para balancear a divisão das classes
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

# 4. Criar um Pipeline: Combinar o vetorizador e o modelo
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', MultinomialNB())
])

# 5. Treinar o modelo
print("Treinando o modelo...")
pipeline.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# 6. Avaliar a performance do modelo (opcional, mas recomendado)
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAcurácia do modelo: {accuracy:.2f}")

# 7. Fazer previsões com novos textos
novo_texto = "A experiência foi a melhor que eu já tive, excelente ."
previsao = pipeline.predict([novo_texto])

print(f"\nTexto: '{novo_texto}'")
print(f"Previsão de sentimento: {previsao[0].upper()}")
