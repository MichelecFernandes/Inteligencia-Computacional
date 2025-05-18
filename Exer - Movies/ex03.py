import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo CSV
df = pd.read_csv('movies.csv')

# Renomear as colunas para português
df = df.rename(columns={
    'movieId': 'id_filme',
    'title': 'titulo',
    'genres': 'generos'
})

# Transformar a coluna 'generos' em várias colunas binárias (0 ou 1)
generos_dummy = df['generos'].str.get_dummies(sep='|')

# Concatenar o DataFrame original com o novo DataFrame de gêneros
df_final = pd.concat([df[['id_filme', 'titulo']], generos_dummy], axis=1)
print(df_final.head())

# Escalar (reescalonar) os dados dos gêneros
scaler = StandardScaler()
generos_escalados = scaler.fit_transform(generos_dummy)

# Converter de volta para um DataFrame para visualizar
generos_escalados_df = pd.DataFrame(
    generos_escalados,
    columns=generos_dummy.columns
)

# Concatenar novamente com id_filme e titulo
df_final_escalado = pd.concat([df[['id_filme', 'titulo']], generos_escalados_df], axis=1)
print(df_final_escalado.head())

kmeans = KMeans(n_clusters=3, random_state=42)
grupos = kmeans.fit_predict(generos_escalados)

# Adicionar o grupo no DataFrame
df['grupo'] = grupos

# Visualizar o número de filmes por grupo
print(df['grupo'].value_counts())

# Plotar os centróides (média dos grupos) para entender quais gêneros dominam em cada grupo
centroides = kmeans.cluster_centers_

# Gerar um gráfico para cada grupo
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

generos = generos_dummy.columns

for i in range(3):
    axes[i].bar(generos, centroides[i])
    axes[i].set_title(f'Grupo {i}')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Reduzir para 2 dimensões com TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=3000)
tsne_result = tsne.fit_transform(generos_escalados)

# Adicionar os resultados TSNE ao DataFrame
df['tsne-2d-one'] = tsne_result[:, 0]
df['tsne-2d-two'] = tsne_result[:, 1]

# Plotar o gráfico de dispersão
plt.figure(figsize=(12,8))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="grupo",
    palette=sns.color_palette("hsv", 3),
    data=df,
    legend="full",
    alpha=0.7
)
plt.title('Gráfico de Dispersão dos Grupos de Filmes (TSNE)')
plt.show()


