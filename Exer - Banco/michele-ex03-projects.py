import pandas as pd
from matplotlib import pyplot as plt
#A sns tem mais gráficos do que a matplotlib
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as ls
from sklearn.svm import SVC as svc
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.dummy import DummyClassifier as dc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LinearRegression as linear_reg
from sklearn.linear_model import LogisticRegression as logist_reg
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.tree import DecisionTreeClassifier as dtc


#Ler o arquivo -> df quer dizer Dataframe(onde é trabalhado os dados)
df = pd.read_csv('projects.csv')
print(df)


"""
Pre-processamento parte 1
"""
# - Apenas para ajustar os valores dos dados
inverte_classe = {
    0: 1,
    1: 0
}

# - Peguei meu dataframe, corrigi o nome da coluna e usei a invertida de valores para melhor visualização
df["finalizado"] = df["unfinished"].map(inverte_classe)

# - drop é usado para apagar a coluna do dataframe -> axis quer dizer coluna
df = df.drop(["unfinished"], axis=1)

# - Renomeando as colunas
df.columns = ["horas_estimadas", "preco", "finalizado"]


# - Describe utilizado para gerar estatisticas
print(df.describe())

""""
    VISUALIZAÇÃO DOS DADOS
"""
# -  SCATTER - Distribuição dos dados por 2 dimensões
# - hue - define a cor dos pontos do gráfico com base em uma variável categórica.
# - scatterplot - de dispersão
# sns.scatterplot(
#     data = df, 
#     x = "horas_estimadas", 
#     y = "preco", 
#     hue = "finalizado")
# plt.show()



# # - RELPLOT - Disperção que separa as classes
# # - A col deve ser igual ao hue
# sns.relplot(
#     data=df, 
#     x = "horas_estimadas", 
#     y = "preco", 
#     col = "finalizado", 
#     hue = "finalizado")
# plt.show()


# # - BARPLOT - Gráfico para contagem de categóricos
# # - Grafico de barras e pizza
# df_contagem_de_finalizado_para_barplot = df["finalizado"].value_counts().to_frame()
# print(df_contagem_de_finalizado_para_barplot)

# # - Gerar grafico de barras
# sns.barplot(
#     data=df_contagem_de_finalizado_para_barplot, 
#     x ="finalizado", 
#     y ="count", 
#     hue ="finalizado")
# plt.show()


# # - Gerar grafico de pizza(Tenho que alterar para ******** PLT ********)
# # -     autopct='%1.1f%%' -> mostrar a porcentagem
# # -    startangle = 0 -> iniciar o gráfico em 90°
# plt.pie(
#     df_contagem_de_finalizado_para_barplot["count"], 
#     labels = ["1", "0"],
#     autopct = '%1.1f%%', 
#     startangle = 90)
# plt.show()


# #Gráficos que exibem estatísticas de atributos numéricos 
# #HISTOGRAMA 
# sns.histplot(
#     df["horas_estimadas"])
# plt.show()

# #Kde plota uma linha do padrão de distibuição
# sns.histplot(
#     df["preco"], 
#     kde = True)
# plt.show()


# #BOXPLOT
# sns.boxplot(
#     data = df,
#     x = df["horas_estimadas"])
# plt.show()

# sns.boxplot(
#     data = df,
#     x = df["preco"])
# plt.show()


"""
   DIVISÃO DE AMOSTRAS E TRANSFORMAÇÃ0 DE DADOS
"""
# Os dados são númericos e possuem escalas diferentes
# Divisão de amostras e normalização
x = df.drop( 
    ["finalizado"], 
    axis = 1)
# X esta excluindo a coluna Y
y = df["finalizado"]
print(x)

# Normalizar significa colocar na mesma escala ou proporção
# Normalizando x em 0 ou 1
scaler = MinMaxScaler()
scale_x = scaler.fit_transform(x)
print("x normalizado: ", scale_x)


# Divisão de amostras
# train_test_split é um metodo da classe modelSelect permite manter a proporcionalidade de classes
train_x, test_x, train_y, test_y = train_test_split(
    scale_x, 
    y,
    test_size = 0.3,
    random_state = 42, 
    stratify = y
)

"""
    TREINO DE MODELOS
"""

# Algoritmo que regra modelo linear
# Meu modelo que estou usando como treino
# clf = ls()
# clf.fit(train_x, train_y)


# """
#     AVALIAÇÃO DE MODELO
# """
# print("Quantidade de testes: ", test_x.shape)
# pred_y = clf.predict(test_x)

# acuracia = accuracy_score(test_y, pred_y)
# acuracia_porcentagem = acuracia * 100
# print("Acurácia LinearSVC: {:.2f}".format(acuracia_porcentagem), "%")


# matriz_confusao = confusion_matrix(test_y, pred_y)
# print("Matriz de confusao: \n", matriz_confusao)

# exit()
# print(test_y.value_counts())

# exit()




# ------------
# Treino usando KNN - 77,47%
# clf = knc()
# clf.fit(train_x, train_y)

# """
#     AVALIAÇÃO DE MODELO
# """
# print("Quantidade de testes: ", test_x.shape)
# pred_y = clf.predict(test_x)

# acuracia = accuracy_score(test_y, pred_y)
# acuracia_porcentagem = acuracia * 100
# print("Acurácia KNN: {:.2f}".format(acuracia_porcentagem), "%")


# matriz_confusao = confusion_matrix(test_y, pred_y)
# print("Matriz de confusao: \n", matriz_confusao)

# exit()
# print(test_y.value_counts())

# exit()


# Treino usando RFC - 78,24%
clf = rfc()
clf.fit(train_x, train_y)

"""
    AVALIAÇÃO DE MODELO
"""
print("Quantidade de testes: ", test_x.shape)
pred_y = clf.predict(test_x)

acuracia = accuracy_score(test_y, pred_y)
acuracia_porcentagem = acuracia * 100
print("Acurácia RFC: {:.2f}".format(acuracia_porcentagem), "%")


matriz_confusao = confusion_matrix(test_y, pred_y)
print("Matriz de confusao: \n", matriz_confusao)

exit()
print(test_y.value_counts())

exit()




# Ele aceitou 397 de 648 que é 61% de 100%


# print("treino y: ", train_y.value_counts())
# print("teste y: ", test_y.value_counts())


# MATRIZ de correlação dos atributos - Quando o valor de x aumenta 1, o valor de y aumenta 1
# # Atributos que tem correlação positiva, você pode excluir 1 dos 2
# # corr é um metodo da classe dataframe que gera uma matriz de correlação dos atributos
# # quando a correlação é maior que 0, ou seja positiva, isso significa que quando 1 valor aumentar o outro aumenta

# correlacao = df.drop(
#     ["finalizado"],
#     axis = 1).corr()

# print(correlacao)

# sns.heatmap(correlacao)
# plt.show()

# exit()


