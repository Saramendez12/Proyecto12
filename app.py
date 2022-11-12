import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.graph_objs as go


pca_3=pd.read_csv("./pca_3.csv")
tabla=pd.read_csv("./tabla.csv")
kmea=pd.read_csv("./kmea.csv")

st.title("Proyecto 12")
st.write("En la presente pagina, se realiza el despliegue del proyecto que buscaba construir un dataset con variables extraidas de la API del Banco Mundial que puedan explicar la felicidad de un país a partir de 10 diferentes variables de 60 paises,para el año 2018")
st.subheader("EXPLICACIÓN VARIABLES USADAS")
st.write("1 Inflación, precios al consumidor (% anual):Cambio porcentual anual en el costo promedio de adquirir una canasta de bienes y servicios")
st.write("2 Indice de Desarrollo Humano:Calcula las contribuciones de la salud y la educación a la productividad del trabajador. El puntaje del índice varía de 0 a 1")
st.write("3 PIB per capita:Es el producto interno bruto dividido por la población a mitad de año.")
st.write("4 Gasto Publico en Educación (% del PIB):El gasto del gobierno general en educación (corriente, capital y transferencias) se expresa como porcentaje del PIB.")
st.write("5 Tasa de suicidios:Es el número de muertes por suicidio en un año por cada 100.000 habitantes.")
st.write("6 Salario medio de los empleados:Remuneración de los trabajadores con contratos explicitos o implicitos.")
st.write("7 Desempleo total:Se refiere a la proporción de la fuerza laboral que no tiene trabajo pero que está disponible y buscando empleo.")
st.write("8 Esperanza de vida: Indica el número estimado de años que viviría un recién nacido")
st.write("9 Poblacion mayor a los 65 años:Porcentaje de la población total que es mayor a 65 años.")
st.write("10 Tasa de homicidios intencionales:Son estimaciones de homicidios ilegales infligidos")

st.subheader("Reducción de dimensionalidad (Algoritmo usado PCA): Este metodo se realiza a través de una descomposición de la varianza")
st.write(" Para hacer uso de este algoritmo hicimos el siguiente paso a paso")
st.write("Estandarización del dataset-----")

Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = 'green', size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()
st.plotly_chart(fig,use_container_widht=True)
st.caption("Grafica interactiva de PCA")
st.subheader("Calificación Davies-Boluldin:")
st.write("Para analizar el numero de clusters optimos realizamos el grafico del codo, con un rango de 2 a 10. En este se busca identificar la cantidad optima de clusters que minimizen la puntación de Davies Bouldin. A pesar de en que este grafico las medidas no se llevan mucha diferencia solo de 1 en 1 , se ve claramente que el valor minimo es 3, por lo tanto se concluye que la puntuación de Davies Bouldin se minimiza con 3 grupos o clusters y se puede considera este k means. ")
st.write("AQUI AÑADIR EL CODITOOOOOOOO")
st.write("Por ultimo la calificación o puntaje de Davies Bouldin es de 1.034 aproximadamente el cual se consigue con el número de grupos ya elegido.")

st.subheader("Algoritmo de reducción de dimensionalidad seleccionado PCA:")
st.code(""" X = tabla.to_numpy()
scal = StandardScaler()
X_scal = scal.fit_transform(tabla)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scal)

tabla = tabla.to_numpy()
#Estandarización del Dataset (tabla) 
l = []
for i in tabla.T:
  u = i.mean()
  s = i.std()
  scal = (i - u) / s
  l.append(scal)

tabla_scal = np.array(l).T
cov_x = np.cov(tabla_scal.T)
cov_x
np.linalg.eig(pd.DataFrame(tabla).corr().to_numpy())
val_p, vec_p = linalg.eig(cov_x)
val_p, vec_p

val_p = val_p[:3]
vec_p = vec_p[:, :3]
W = vec_p
W
pca_p = tabla_scal @ W
pca_p = pd.DataFrame(pca_p, columns=[f'PCA{i}' for i in range(1, pca_p.shape[1] + 1)])
scal = StandardScaler()
tabla_scal = scal.fit_transform(tabla)

pca = PCA(n_components=3)
tabla_pca = pca.fit_transform(tabla_scal)

pca_3 = pd.DataFrame(tabla_pca, columns=['PCA1','PCA2','PCA3'])

z = pca_3['PCA1']
x = pca_3['PCA2']
y = pca_3['PCA3']
 
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, z, y, color = "green")
 
# show plot
plt.show()""",language="python")
             
st.subheader("Algoritmo de Clustering seleccionado Kmeans:")
st.code("""kmeans = KMeans(n_clusters=3, random_state=777,algorithm='elkan').fit(X_scal)
pca_3['labels'] = kmeans.labels_

centroides = {}
k = 3
for i in range(k):
  centroides[i] = X[np.random.choice(len(X))]
centroides

def dista_euclidiana(puntos, centroide):
  return np.sqrt(sum((puntos-centroide)**2))

distancias = {}

for i in range(len(X)):
  distancias[i] = []

for pos, dato in enumerate(X):
  for pos_, centroide in centroides.items():
    distancias[pos].append(dista_euclidiana(dato, centroide))
puntos_centroides = {}

for i in range(k):
  puntos_centroides[i] = []

for pos, dists in distancias.items():
  puntos_centroides[dists.index(min(dists))].append(X[pos])

k = 3
centroides = {}
iteraciones = 6
contador = 12
for i in range(k):
  centroides[i] = X[np.random.choice(len(X))]

for itera in range(iteraciones):
  #distancias

  distancias = {}

  for pos, datos in enumerate(X):
    distancias[pos] = []
    for pos_, centroide in centroides.items():
      distancias[pos].append(dista_euclidiana(datos, centroide))

  #asignar cada punto a un centroide por la mínima distancia

  puntos_centroides = {}

  for i in range(k):
    puntos_centroides[i] = []

  for pos_dato, distancias in distancias.items():
    puntos_centroides[distancias.index(min(distancias))].append(X[pos_dato])

  #nuevo centroide
  fig, ax = plt.subplots(1, 1, figsize=(10,6))
  for centroide, datos in centroides.items():
    ax.scatter(np.vstack(puntos_centroides[centroide])[:,0],np.vstack(puntos_centroides[centroide])[:,1])
    ax.scatter(datos[0],datos[1], marker='x', s=400, color='k')
  
  fig.savefig(f'imagen{contador}.png')
  plt.show()


  for centroide, datos in puntos_centroides.items():
    centroides[centroide] = np.average(np.vstack(datos), axis=0)

  contador += 1""",language="python")

st.subheader("Concluya sobre los clústers de manera descriptiva y gráfica.")
X = tabla.to_numpy()
scal = StandardScaler()
X_scal = scal.fit_transform(tabla)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scal)


pca_3['labels'] = kmea
Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
labels = kmea
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

st.plotly_chart(fig,use_container_widht=True)

st.subheader("Análisis descriptivo y gráfico")
st.subheader("Conclusión General")
st.write("[Link al Notebook](https://colab.research.google.com/drive/115jkwsUACKRFmJAgcehE8PxzRzFha0Y1?usp=sharing)")



