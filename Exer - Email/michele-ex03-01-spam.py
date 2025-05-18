import pandas as pd
import numpy 
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# Gerar duas colunas de valores inteiros aleatórios
col1 = numpy.random.randint(0, 100, 1000)
col2 = numpy.random.randint(0, 100, 1000)

# Gerar a coluna binária com 95% de 0s e 5% de 1s
col3 = numpy.random.choice([0, 1], 
                        size = 1000, 
                        p=[0.95, 0.05])

# Criar o DataFrame
df = pd.DataFrame({'Dados X': col1, 'Dados Y': col2, 'Spam ou não': col3})

print(df) # Exibir as primeiras linhas do DataFrame

# Apagar a coluna Spam ou não
coluna_x = df.drop(["Spam ou não"], axis=1)
coluna_y = df["Spam ou não"]


# Fazer a divisão para o treino de 70% dos dados e 30% de teste
train_x, test_x, train_y, test_y = train_test_split(coluna_x, coluna_y, test_size=0.3, random_state=42, stratify=coluna_y) 

classificacao = DummyClassifier(strategy='constant', constant=0)
classificacao.fit(train_x, train_y)
previsao_y = classificacao.predict(test_x)
print(previsao_y)

# Avalidar acurácia
acuracia = accuracy_score(test_y, previsao_y)
print("Valor de acurácia: {}".format(acuracia))

matriz_confusao = confusion_matrix(test_y, previsao_y)
print("Resultado da matriz de Confusão: ")
print(matriz_confusao)

f1 = f1_score(test_y, previsao_y)
print("F1: ") # verifica que o modelo foi chute
print(f1)

# Obter probabilidades previstas
y_prob = classificacao.predict_proba(test_x)[:, 1]  # Pegamos apenas a probabilidade de ser 1

# Calcular a curva ROC
fpr, tpr, _ = roc_curve(test_y, y_prob)
auc = roc_auc_score(test_y, y_prob)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, 
         tpr, 
         color = 'red', 
         label = 'ROC curve (área = {:.2f})'.format(auc))

plt.plot([0, 1], 
         [0, 1], 
         color = 'green', 
         linestyle='--')

# Limite dos eixo x
plt.xlim([0.0,
           1.0])

# Limite dos eixo y
plt.ylim([0.0,
           1.05])

plt.xlabel('Taxa de falsos positivos (FPR): ')

plt.ylabel('Taxa de verdadeiros positivos (TPR): ')


# Título do gráfico
plt.title('Curva ROC')

# Legenda do gráfico
plt.legend(loc='lower right')

plt.grid()
plt.show()