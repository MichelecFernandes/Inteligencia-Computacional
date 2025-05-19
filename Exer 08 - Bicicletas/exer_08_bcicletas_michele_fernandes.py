# Michele Cristina Fernandes 1202210061

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('Bicicletas.csv')

# 2)
# Normalização dos dados
scaler = MinMaxScaler()
data[['temperatura']] = scaler.fit_transform(data[['temperatura']])

# Divisão em treino e teste
X = data[['temperatura', 'clima']]
y = data['bicicletas_alugadas']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# 4) 
# Modelo de regressão - Árvore de Decisão
def arvoce_Decision():
    modelo_arvore = DecisionTreeRegressor(random_state=42)
    modelo_arvore.fit(train_X, train_y)
    pred_arvore = modelo_arvore.predict(test_X)
    print("Modelo árvore de decisão (Regressão):", pred_arvore, "\n")


def modeloKnn():
    clf = SVR()
    clf.fit(train_X, train_y)
    print("Quantidade de testes: ", test_X.shape)
    pred_y = clf.predict(test_X)

    mse = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    print("Erro médio entre os valores reais: {:.2f}".format(mse))
    print("Coeficiente de determinação: {:.2f}".format(r2), "\n")

# Funções para gerar os gráficos
def gerar_scatterplot():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='temperatura', y='bicicletas_alugadas', hue='clima', data=data)
    plt.title('Temperatura x Bicicletas Alugadas')
    plt.show()

def gerar_histograma():
    plt.figure(figsize=(6, 4))
    sns.histplot(data['bicicletas_alugadas'], bins=20, kde=True)
    plt.title('Bicicletas Alugadas')
    plt.show()

def gerar_boxplot():
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='clima', y='bicicletas_alugadas', data=data)
    plt.title('Clima x Bicicletas Alugadas')
    plt.show()

# 4)
def gerar_grafico_regressao():
    modelo = DecisionTreeRegressor()
    modelo.fit(train_X, train_y)
    pred_y = modelo.predict(test_X)
    plt.figure(figsize=(6, 4))
    plt.scatter(test_X['temperatura'], test_y, color='blue', label='Dados Reais')
    plt.scatter(test_X['temperatura'], pred_y, color='red', alpha=0.5, label='Previsões do Modelo')
    plt.title('Previsão de Bicicletas Alugadas')
    plt.xlabel('Temperatura')
    plt.ylabel('Bicicletas Alugadas')
    plt.legend()
    plt.show()
    

# Função do Menu
def menu():
    opcao = -1
    while opcao != 0:
        print("Escolha um gráfico para gerar:")
        print("1 - Scatter Plot")
        print("2 - Histograma")
        print("3 - BoxPlot")
        print("4 - Gráfico de Regressão")
        print("5 - Arvore de decisão")
        print("6 - Knn")
        print("0 - Sair")

        opcao = int(input("Digite a opção desejada: "))
        
        if opcao == 1:
            gerar_scatterplot()
        elif opcao == 2:
            gerar_histograma()
        elif opcao == 3:
            gerar_boxplot()
        elif opcao == 4:
            gerar_grafico_regressao()
        elif opcao == 5:
            arvoce_Decision()
        elif opcao == 6:
            modeloKnn()
        elif opcao == 0:
            print("Saindo...")
        else:
            print("Opção inválida!")

# Executa o menu
menu()

