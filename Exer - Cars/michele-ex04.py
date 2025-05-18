# Michele Cristina Fernandes 1202210061

# EXERCÍCIO PRÁTICO DE APRENDIZADO SUPERVISIONADO CLASSIFICAÇÃO

# mileage_per_year: indica a quantidade de milhas que o carro rodou por ano desde sua
# fabricação.
# • model_year: indica o ano de fabricação.
# • price: indica o preço.
# • sold: Indica se o carro foi vendido ou não (yes / no).

# 1) Crie um programa em Python de aprendizado de máquinaque possa ler os dados do arquivo CSV, treinar e aprender com base nos registros e testar possíveis classificações de novos carros que poderão ser vendidos ou não.
# O QUE DEVE SER FEITO:

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# • ETAPA 1: PRÉ-PROCESSAMENTO DOS DADOS - TRANSFORMAÇÃO –
# PARTE 1

dataframe = pd.read_csv('car-prices.csv')

# 2. Sold: Mapeie a coluna sold para uma nova coluna com valores 1(yes) ou 0(no).
inverte_tipo_classe = {
    'yes': 1,
    'no': 0
}

# 1. Renomeie as colunas para português utilizando o rename do Pandas.
dataframe["km_rodados"] = dataframe["mileage_per_year"]

# 3. Model_year: Ao invés de usar a dimensão ano do modelo, crie uma nova coluna que armazene a idade do veículo (Ano atual - ano do modelo).
dataframe["idade_veiculo"] = 2025 - dataframe["model_year"]

dataframe["preco"] = dataframe["price"]
dataframe["vendido?"] = dataframe["sold"].map(inverte_tipo_classe)

# 4. Deixe o dataframe somente com as colunas (milhas_por_ano, idade, preco, vendido). Delete as demais colunas.
dataframe = dataframe.drop(["id"], axis=1)
dataframe = dataframe.drop(["mileage_per_year"], axis=1)
dataframe = dataframe.drop(["model_year"], axis=1)
dataframe = dataframe.drop(["price"], axis=1)
dataframe = dataframe.drop(["sold"], axis=1)

print(dataframe)

# • ETAPA 2: VISUALIZAÇÃO DOS DADOS
# 1. Plote um gráfico de dispersão dos dados – verifique uma forma de exibir 3 dimensões.

sns.scatterplot(data=dataframe, x='km_rodados', y='preco', hue='idade_veiculo')
plt.title('Relação entre Km Rodados e Preço dos Veículos')
plt.show()

# 1. Responda: É possível um modelo linear resolver o problema?
#      R: Não.
# 2. Responda: Há ruídos?
#      R: Sim, muito das classes estão sobrepondo outras.


# 2. Plote um gráfico de pizza para a contagem de exemplos das classes.
df_contagem_vendidos = dataframe["vendido?"].value_counts().to_frame()
print(df_contagem_vendidos)

plt.pie(
    df_contagem_vendidos["count"],
    labels = ["1", "0"],
    autopct = '%1.1f%%', 
    startangle = 90)
plt.title("Contagem de Veículos Vendidos")
plt.legend(["Vendido (1)", "Não Vendido (0)"], 
           loc="upper right") 
plt.show()

# 1. Responda: Há desbalanceamento de classes?
#      R: Sim, há uma diferença entre as classes. A classe 0 representa 42% e a classe 1 representa 58%.
 
# 3. Plote histogramas e gráfico de caixas para analisar os atributos numéricos.

## Histogramas 
    # Kw rodados 
sns.histplot(
    dataframe["km_rodados"], 
    kde = True)
plt.title("Histograma por Kw rodados")
plt.ylabel("Quantidade")
plt.xlabel("Kw rodados")
plt.show()

    # Idade veículo
sns.histplot(
    dataframe["idade_veiculo"], 
    kde = True)
plt.title("Histograma por Idade veículo")
plt.ylabel("Quantidade")
plt.xlabel("Idade veículo")
plt.show()

    # Preço 
sns.histplot(
    dataframe["preco"], 
    kde = True)
plt.title("Histograma por Preço")
plt.ylabel("Quantidade")
plt.xlabel("Preço R$")
plt.show()

    # Vendido 
sns.histplot(
    dataframe["vendido?"], 
    kde = True)
plt.title("Histograma por vendas")
plt.ylabel("Quantidade")
plt.xlabel("Vendido?")
plt.show()


## Gráfico de caixas
    # Kw rodados 
sns.boxplot(
    data = dataframe,
    x = dataframe["km_rodados"])
plt.title("Grafico de caixas por km rodados")
plt.show()

    # Idade veículo
sns.boxplot(
    data = dataframe,
    x = dataframe["idade_veiculo"])
plt.title("Grafico de caixas por idade do veículo")
plt.show()

    # Preço 
sns.boxplot(
    data = dataframe,
    x = dataframe["preco"])
plt.title("Grafico de caixas por preço")
plt.show()

    # Vendido 
sns.boxplot(
    data = dataframe,
    x = dataframe["vendido?"])
plt.title("Grafico de caixas para verificar se foi vendido ou não")
plt.show()


# 1. Responda: Há atributos com outliers?
    # R: Sim, existem atributos com outliners, com valores altos, fora do padrão . Isso foi visto nos gráficos de kw rodados, idade do veículo e preço.
# 2. Responda: Há diferenças de escalas entre os atributos?
    # R: Sim, tem atributos que atingem mais de 4 digitos, como o km rodados, e tem atributos do tipo se ele foi vendido que é apenas 0 e 1.

# 4. Plote um gráfico de calor para verificar a correlação dos atributos.
# 1. Responda: É possível remover algum atributo?
    # R: Sim, é possível remover os atributos de km_rodaods e idade_veiculo. Já o preço apresenta maior correlacao com o atributo vendido.

correlacao = dataframe.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            linewidths=0.5)
plt.title('Mapa de calor para verificar a correlação dos atributos')
plt.show()

# • ETAPA 3 – DIVISÃO DE AMOSTRAS – PARTE 1
# 1. Divida o dataframe em X(atributos) e Y(classes).

# x sao os atributos
x = dataframe[["km_rodados", "idade_veiculo", "preco"]]

# y eh a classe
y = dataframe["vendido?"]

# • ETAPA 4 – PRÉ-PROCESSAMENTO DOS DADOS – P TRANSFORMAÇÃO - PARTE 2
# 1. Se houver atributos alfanuméricos, efetue a transformação utilizando a técnica – One Hot Encoding ou Label Encoder.

# 2. Normalize os dados dos atributos utilizando a abordagem – padrão 0 / 1. Armazene em um atributo específico.
# 	Normalizar entre 0 e 1
scaler = MinMaxScaler()
scale_x_normalizado = scaler.fit_transform(x)
print("x normalizad entre 0 e 1: ", scale_x_normalizado)

# 3. Normalize os dados dos atributos utilizando a abordagem – Z score. Armazene em um atributo específico.
# Padronizar com média 0 e desvio padrão 1
scaler_zscore = StandardScaler()
x_zscore = scaler_zscore.fit_transform(x)
print("x normalizado com media 0 e 1: ", x_zscore)

# • ETAPA 5 – DIVISÃO DE AMOSTRAS
# 1. Utilizando a técnica Hold Out, efetue a divisão dos dados em uma amostra para
# TREINO o outra para TESTE. Utilize a proporção 70 / 30. Estratifique as classes.
train_x, test_x, train_y, test_y = train_test_split(
    scale_x_normalizado, 
    y,
    test_size = 0.3,
    random_state = 42, 
    stratify = y
)
# 1. Responda: Com base nas respostas da ETAPA 2, você vai utilizar todo o
# conjunto de dados para treinar modelos de aprendizado de máquina?
 # R: Nao, não irei usar todo o conjunto, visto que ha uma grande diferença de de padrões dos valores de acordo com cada atributo. 


# • ETAPA 6 – TREINAMENTO DO MODELO
# 1. Efetue treino com um modelo linear
clf = LinearSVC()
clf.fit(train_x, train_y)
print("Quantidade de testes: ", test_x.shape)
pred_y = clf.predict(test_x)

# 2. Efetue treino com o KNN
acuracia = accuracy_score(test_y, pred_y)
acuracia_porcentagem = acuracia * 100
print("Acurácia KNN: {:.2f}".format(acuracia_porcentagem), "%")

# 3. Efetue treino com árvore de decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(train_x, train_y)
pred_arvore = modelo_arvore.predict(test_x)
print("Modelo arvore de decisão:", pred_arvore)

# • ETAPA 7 – AVALIAÇÃO DOS MODELOS
# ◦ Para cada modelo treinado acima, efetue testes com a massa de testes e avalie:
# 1. Acurácia
# 2. Matriz de Confusão
# 3. F1 – Score
# 4. Plote a Curva ROC

clf = DummyClassifier(strategy="constant", constant=0)
clf.fit(train_x, train_y)

pred_y = clf.predict(test_x)

acuracia = accuracy_score(test_y, pred_y)
matriz_confusao = confusion_matrix(test_y, pred_y)
f1 = f1_score(test_y, pred_y)
roc = roc_auc_score(test_y, pred_y)

print("Matriz de Confusão:\n {}".format(matriz_confusao))
print("F1: {}".format(f1))
print("ROC: {}".format(roc))


# Obtendo probabilidades preditas para a classe positiva
y_prob = clf.predict_proba(test_x)[:, 1]

# Criando a curva ROC
fpr, tpr, _ = roc_curve(test_y, y_prob)
roc_auc = auc(fpr, tpr)

# Plotando
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
# 1. Responda: Qual foi o melhor modelo para se colocar em produção?
 # R: O KNN, pois sua acurácia foi de 70.37%