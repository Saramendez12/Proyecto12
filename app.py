import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

pca_3=pd.read_csv("./pca_3.csv")
tabla=pd.read_csv("./tabla.csv")
kmea=pd.read_csv("./kmea.csv")

st.title("Proyecto 12")
st.write("En la presente pagina, se realiza el despliegue del proyecto que buscaba construir un dataset con variables extraidas de la API del Banco Mundial" 
"que puedan explicar la felicidad de un país a partir de 10 diferentes variables de 60 paises,para el año 2018")
st.subheader("Explicación de variables usadas")
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

st.code(""" X = tabla.to_numpy()
scal = StandardScaler()
X_scal = scal.fit_transform(tabla)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scal)

tabla = tabla.to_numpy()
#Estandarización del Dataset (tabla) para hacer más sencillo su uso. 
l = []
for i in tabla.T:
  u = i.mean()
  s = i.std()
  scal = (i - u) / s
  l.append(scal)

tabla_scal = np.array(l).T
#Construcción de las matrices de varianzas y covarianzas que nos da nuestra tabla estandarizada 
cov_x = np.cov(tabla_scal.T)
cov_x
#Uso de la libreria Linalg para matrices y vectores de numpy.
np.linalg.eig(pd.DataFrame(tabla).corr().to_numpy())
#Descomposición de la varianza
val_p, vec_p = linalg.eig(cov_x)
val_p, vec_p
#Selecciones de 3 componentes a partir de la descomposición
val_p = val_p[:3]
vec_p = vec_p[:, :3]
#Creación de una matriz de proyección
W = vec_p
W
#Proyección de X nuestra tabla estandarizada en  la matriz de proyección
pca_p = tabla_scal @ W
pca_p = pd.DataFrame(pca_p, columns=[f'PCA{i}' for i in range(1, pca_p.shape[1] + 1)])
#Ajuste de nuestra X_scal que nos permite tener el Dataset pca3 para nuestro componentes principales seleccionados.
scal = StandardScaler()
tabla_scal = scal.fit_transform(tabla)

pca = PCA(n_components=3)
tabla_pca = pca.fit_transform(tabla_scal)
# Creación de Grafica 3D no interactiva para visualizar los componentes.
pca_3 = pd.DataFrame(tabla_pca, columns=['PCA1','PCA2','PCA3'])

z = pca_3['PCA1']
x = pca_3['PCA2']
y = pca_3['PCA3']

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(x, z, y, color = "green")

plt.show()""",language="python")

st.write("Visualización de la reducción de dimensionalidad por componentes"
         "principales en 3D Interactiva")
Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = 'green', size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()
st.plotly_chart(fig,use_container_widht=True)
st.caption("Grafica interactiva de PCA")

st.write("Para poder graficar y ver como se explica la varianza a traves de los componentes principales seleccionados, realizamos a partir del pca_3 (dataframe de los PCA)"
          "la creación de la varianza explicada y acumulada, en el cual se ve de forma clara que el numero de componentes elegido (3) es el que puede de mejor manera explicar la varianza")
st.image("./Imagenes12/VAR.PNG")

st.subheader("Calificación Davies-Bouldin:")
st.write("Para analizar el numero de clusters optimos realizamos el grafico del codo, con un rango de 2 a 10. En este se busca identificar" 
"la cantidad optima de clusters que minimizen la puntación de Davies Bouldin. A pesar de en que este grafico las medidas no se llevan mucha"
"diferencia solo de 1 en 1 , se ve claramente que el valor minimo es 3, por lo tanto se concluye que la puntuación de Davies Bouldin se minimiza"
         "con 3 grupos o clusters y se puede considera este k means. ")
st.image("./Imagenes12/DB.PNG")
st.write("Por ultimo la calificación o puntaje de Davies Bouldin es de 1.034 aproximadamente el cual se consigue con el número de grupos ya elegido.")

st.subheader("Algoritmo de Clustering seleccionado K means:")
st.code("""kmeans = KMeans(n_clusters=3, random_state=777,algorithm='elkan').fit(X_scal)
pca_3['labels'] = kmeans.labels_

#Elección de los centroides a partir de los k ya elegidos.
centroides = {}
k = 3
for i in range(k):
  centroides[i] = X[np.random.choice(len(X))]
centroides

#Función para calcular la distancia euclidiana
def dista_euclidiana(puntos, centroide):
  return np.sqrt(sum((puntos-centroide)**2))
distancias = {}
for i in range(len(X)):
  distancias[i] = []
for pos, dato in enumerate(X):
  for pos_, centroide in centroides.items():
    distancias[pos].append(dista_euclidiana(dato, centroide))
    
#Asignación de cada cluster seleccionado (1,2,3) por la mínima distancia.
puntos_centroides = {}
for i in range(k):
  puntos_centroides[i] = []
for pos, dists in distancias.items():
  puntos_centroides[dists.index(min(dists))].append(X[pos])
  
#Inicio de la graficación de los centroides con determinados clusters.
k = 3
centroides = {}
iteraciones = 6
contador = 12
for i in range(k):
  centroides[i] = X[np.random.choice(len(X))]
for itera in range(iteraciones):
  distancias = {}
  for pos, datos in enumerate(X):
    distancias[pos] = []
    for pos_, centroide in centroides.items():
      distancias[pos].append(dista_euclidiana(datos, centroide))
  puntos_centroides = {}
  for i in range(k):
    puntos_centroides[i] = []
  for pos_dato, distancias in distancias.items():
    puntos_centroides[distancias.index(min(distancias))].append(X[pos_dato])
  fig, ax = plt.subplots(1, 1, figsize=(10,6))
  for centroide, datos in centroides.items():
    ax.scatter(np.vstack(puntos_centroides[centroide])[:,0],np.vstack(puntos_centroides[centroide])[:,1])
    ax.scatter(datos[0],datos[1], marker='x', s=400, color='k')
  fig.savefig(f'imagen{contador}.png')
  plt.show()
  for centroide, datos in puntos_centroides.items():
    centroides[centroide] = np.average(np.vstack(datos), axis=0)
  contador += 1""",language="python")

st.image("./Imagenes12/Visualización Centroides y Clusters.gif")

st.subheader("Concluya sobre los clústers de manera descriptiva y gráfica.")
st.write("Se hizo la elección del algoritmo de clasificación kmeans, este logro agrupar los datos (objetos) en k grupos, para este caso a partir" 
"del puntaje de Davies Boudin se tomo 3 grupos basandose así en sus caractersticas en común. El agrupamiento se realizo a partir de la formula que" 
"nos permitia minimizar la sumatoria de las distancias(euclidianas) y los centroides dentro de los k." 
"Para poder ver de forma grafica el movimiento de los objetos(datos) y ubicar los centroides que recojian la mayor información a aprtir de las distancias," 
"se realizo 6 interaciones, a traves de nuestros datos (Tabla o X) y los k=3.Se concluye por lo tanto que graficamente los centroides se ubican en las distancias"
"más cercanas a los grupos, pero que dentro de almenos un grupo el centroide tiene distancias muy alejadas con o de los objetos(datos) del grupo como sucede" 
"con los dos datos atipicos que rozan el punto 30.")

kmean=kmea.to_numpy()
pca_3['labels'] = kmean
Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
labels = kmean
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

st.write("Visualización de la reducción de dimensionalidad por componentes"
         "principales en 3D Interactiva separados por los kmeans elegidos(k=3)")
st.plotly_chart(fig,use_container_widht=True)
st.caption("Grafica interactiva de PCA con los k means presentes")

st.subheader("Conclusión General")
st.write("Se puede concluir primeramente que el algoritmo de componentes principales redujo nuestra dimensionalidad de variables de 10 a 3, mostrando así un buen resultado"
         "que se aprecia visualmente por los graficos, así mismo se puede concluir y analizar que a traves del algoritmo de agrupamiento K means, se agrupo en 3 grupos"
         "despues de poner a prueba el puntaje de Davies - Bouldin, a partir de este algoritmo se pudo evaluar la agrupación de los componentes y como estos podrían"
         "explicar de una forma más sencilla las variables. Visualmente es más posible ver que en cuanto a si buscamos explicar el indice de felicidad creado con las variables"
         "se ve que la mayoria de datos estan en forma agrupada al lado izquierdo y el lado derecho cuenta con uno u otro dato un poco más alejado.")

st.subheader("[Link al Notebook](https://colab.research.google.com/drive/115jkwsUACKRFmJAgcehE8PxzRzFha0Y1?usp=sharing)")



