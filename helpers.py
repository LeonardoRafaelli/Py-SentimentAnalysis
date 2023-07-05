import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# Download do tokenizador (divide o texto em suas palavras)
nltk.download('punkt')
# Download dos StopWords (palavras desnecessárias para a analise, ex. a, de, um)
nltk.download('stopwords')


def preprocess_text(sentence):
    # Remove as pontuações de um texto (susbtitui por "")
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Troca todas as letras para minúscula
    sentence = sentence.lower()
    # Cria um vetor com as palavras do texto
    words = word_tokenize(sentence)
    # Cria um set com as palavras 'stop words' de português
    stop_words = set(stopwords.words('portuguese'))
    # Filtra as palavras com um for, removendo as stop words da frase
    words = [word for word in words if word not in stop_words]
    # Reduz as palavras ao seu radical (andando -> andar)
    stemmer = PorterStemmer()
    # Instancia uma lista de palavras radicalizadas
    words = [stemmer.stem(word) for word in words]
    sentence = ' '.join(words)  # Concatena as frases
    return sentence


train_data = [
    ("Essa comida é deliciosa!", "positivo"),
    ("Nunca vi algo tão ruim como esse filme...", "negativo"),
    ("Estou completamente apaixonado por esse livro!", "positivo"),
    ("O atendimento foi péssimo nesse restaurante...", "negativo"),
    ("Essa música é muito animada, adoro!", "positivo"),
    ("Fiquei extremamente decepcionado com o serviço desse hotel...", "negativo"),
    ("Essa pintura é realmente impressionante!", "positivo"),
    ("Não recomendo esse produto, é de baixa qualidade...", "negativo"),
    ("Adorei o show, foi sensacional!", "positivo"),
    ("O aplicativo é cheio de bugs, não funciona direito...", "negativo"),
    ("Esse livro me fez refletir sobre a vida, muito bom!", "positivo"),
    ("A entrega demorou muito, estou insatisfeito...", "negativo"),
    ("Essa paisagem é deslumbrante, fiquei maravilhado!", "positivo"),
    ("O suporte ao cliente foi terrível, não resolveram meu problema...", "negativo"),
    ("Recomendo esse produto, superou minhas expectativas!", "positivo"),
    ("O filme é tedioso e sem graça, não gostei...", "negativo"),
    ("Esse jogo é viciante, não consigo parar de jogar!", "positivo"),
    ("A embalagem veio danificada, péssimo serviço de entrega...", "negativo"),
    ("Essa peça de teatro é magnífica, vale muito a pena!", "positivo"),
    ("A comida estava fria e sem sabor, não voltarei nesse restaurante...", "negativo"),
    ("Esse álbum é incrível, não consigo parar de ouvir!", "positivo"),
    ("O atendimento ao cliente foi péssimo, não foram educados...", "negativo"),
    ("Adorei a nova coleção de roupas, estão lindas!", "positivo"),
    ("O filme é previsível e mal feito, não recomendo...", "negativo"),
    ("Esse celular tem um desempenho excepcional, estou impressionado!", "positivo"),
    ("O serviço de entrega atrasou e não deram nenhuma explicação...", "negativo"),
    ("Essa peça de arte é única, fiquei encantado!", "positivo"),
    ("A qualidade desse produto é terrível, que desperdício de dinheiro...", "negativo"),
    ("Esse concerto foi incrível, uma experiência inesquecível!", "positivo"),
    ("O software está cheio de erros, não cumpre o que promete...", "negativo"),
    ("Essa viagem foi maravilhosa, conheci lugares incríveis!", "positivo"),
    ("O atendimento no restaurante foi péssimo, fui tratado com desrespeito...", "negativo"),
    ("Adoro essa música, sempre me anima!", "positivo"),
    ("O filme é chato e sem emoção, me arrependi de assistir...", "negativo"),
    ("Esse produto é funcional, cumpre sua função.", "neutro"),
    ("O filme tem uma trama interessante, mas a execução poderia ser melhor.", "neutro"),
    ("A comida estava ok, não era excepcional, mas também não era ruim.", "neutro"),
    ("O serviço de atendimento ao cliente foi mediano, nada fora do comum.", "neutro"),
    ("A música é agradável de se ouvir, não é marcante, mas também não é desagradável.", "neutro"),
    ("O produto apresentou algumas falhas, mas ainda é utilizável.", "neutro"),
    ("O filme possui algumas cenas emocionantes, mas também tem momentos arrastados.", "neutro"),
    ("A comida estava aceitável, não me surpreendeu, mas também não me decepcionou.", "neutro"),
    ("O serviço prestado foi regular, sem grandes destaques positivos ou negativos.", "neutro"),
    ("A música é tranquila e agradável, mas não é memorável.", "neutro"),
    ("O produto é comum, não se destaca em relação aos concorrentes.", "neutro"),
    ("O filme é mediano, não é extraordinário, mas também não é terrível.", "neutro"),
    ("A comida estava satisfatória, nada excepcional, mas também não estava ruim.", "neutro"),
    ("O atendimento ao cliente foi padrão, sem grandes surpresas positivas ou negativas.", "neutro"),
    ("A música é agradável aos ouvidos, mas não tem um impacto duradouro.", "neutro"),
    ("O produto atende às expectativas básicas, sem grandes diferenciais.", "neutro"),
    ("O filme tem uma história interessante, porém a execução deixa a desejar.", "neutro"),
    ("A comida estava regular, não se destacou positivamente, mas também não foi terrível.", "neutro"),
    ("O serviço de entrega foi dentro do prazo, sem problemas ou atrasos.", "neutro"),
    ("A música é ok, não é marcante, mas também não é desagradável de se ouvir.", "neutro"),
    ("O produto é funcional, mas não oferece recursos extras.", "neutro"),
    ("O filme possui alguns momentos cativantes, mas também tem suas falhas.", "neutro"),
    ("A comida estava razoável, não surpreendeu, mas também não decepcionou.", "neutro"),
    ("O atendimento ao cliente foi razoável, não houve grandes problemas, mas também nada excepcional.", "neutro"),
    ("A música é suave e agradável, mas não é memorável.", "neutro"),
    ("O produto é comum, não se destaca positiva ou negativamente.", "neutro"),
    ("O filme é regular, não é excepcional, mas também não é péssimo.", "neutro"),
    ("A comida estava satisfatória, dentro do esperado, sem grandes surpresas.", "neutro"),
    ("O serviço prestado foi mediano, sem grandes destaques positivos ou negativos.", "neutro"),
    ("A música é agradável, mas não se destaca entre outras do mesmo gênero.", "neutro")
]

# TF - Quantidade de ocorrências de uma palavra / Quantidade de palavras no texto
# IDF - Quantidade de documentos no corpus / Quantidade de documentos que possuem a palavra
# TF-IDF = TF * IDF (Para cada texto, haverá um vetor, a palavra será identificada pelo resultado dessa multiplicação)
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
# Vetorizando as frases de treino e enviando para o vectorizer
train_features = vectorizer.fit_transform([x[0] for x in train_data])
# Cria um vetor com os sentimentos de cada um dos textos
train_labels = [x[1] for x in train_data]


# Cria um modelo de classificão linear, com o intuito de encontrar o melhor hiperplano entre as classes
classifier = svm.SVC(kernel='linear')
# Treinar o SVM para maximixar a margem entre as classes (sentimentos) aumentando a precisão
classifier.fit(train_features, train_labels)


# Classifica o sentimento de um texto entre um dos labels entregues
def predict_sentiment(sentence):
    # Realizando o pré processamento do dado
    sentence = preprocess_text(sentence)
    # Vetorizando o texto pré-processado
    features = vectorizer.transform([sentence])
    # Classificando o sentimento de acordo com o texto vetorizado
    sentiment = classifier.predict(features)[0]
    return sentiment
