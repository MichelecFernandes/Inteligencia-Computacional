# Michele Cristina Fernandes 1202210061
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Exercicio 1
# Atividade 1.1
# Lê o arquivo CSV
dataframe = pd.read_csv("movies.csv")

# Atividade 1.2
# Renomeação de colunas
dataframe = dataframe.rename(columns={'genres': 'genero', 'title': 'titulo', 'movieId': 'filmeId'})

# One-Hot Encoding dos gêneros
# método de conversão de variáveis categóricas
covertendo_variaveis = dataframe["genero"].str.get_dummies()

# Escalando os dados
generos_one_hot_encoding_escalaods = StandardScaler().fit_transform(covertendo_variaveis)

# Criando e ajustando modelo KMeans
# Passando hiperparametro
modelo = KMeans(n_clusters=3)
modelo.fit(generos_one_hot_encoding_escalaods)

# Atribuindo os grupos ao DataFrame original
dataframe["grupo"] = modelo.labels_

# Mostrar apenas os filmes que estão no grupo 0
grupo_0 = dataframe[dataframe["grupo"] == 0]

# Mostrar apenas os filmes que estão no grupo 1
grupo_1 = dataframe[dataframe["grupo"] == 1]

# Mostrar apenas os filmes que estão no grupo 2
grupo_2 = dataframe[dataframe["grupo"] == 2]

# Reduz de 20 dimensões para 2 usando TSNE
# TSNE é uma técnica de redução de dimensionalidade, muito usada para visualizar dados complexos
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
generos_reduzidos = tsne.fit_transform(generos_one_hot_encoding_escalaods)

# Cria um DataFrame para visualização
dataframe_tsne = pd.DataFrame()
dataframe_tsne["x"] = generos_reduzidos[:, 0]
dataframe_tsne["y"] = generos_reduzidos[:, 1]
dataframe_tsne["grupo"] = dataframe["grupo"]

# Plotando o gráfico
genero_coluna = "Adventure"
dados_originais = covertendo_variaveis[genero_coluna]
dados_escalonados = StandardScaler().fit_transform(dados_originais.values.reshape(-1, 1)).flatten()


# Plotando o gráfico - Agrupando com o KMeans - Parte 1
generos = covertendo_variaveis.columns
centroides = modelo.cluster_centers_  
for i, centroide in enumerate(centroides):
    plt.figure(figsize=(12, 4))
    plt.bar(generos, centroide, color='blue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# Plotando o gráfico - Agrupando com o KMeans - Parte 2
plt.figure(figsize=(10, 6))
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
for grupo in dataframe_tsne["grupo"].unique():
    grupo_dados = dataframe_tsne[dataframe_tsne["grupo"] == grupo]
    plt.scatter(grupo_dados["x"], grupo_dados["y"], label=f"Grupo {grupo}", alpha=0.6, s=30, color=cores[grupo])

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Gráfico de dispersão dos grupos em 2 dimensões")
plt.show()

