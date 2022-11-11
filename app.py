import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import sklearn.preprocessing 
from sklearn.preprocessing import StandardScaler

pca_3=pd.read_csv("./pca_3.csv")
tabla=pd.read_csv("./tabla.csv")

st.title("Proyecto 12")
st.write(" En la presente pagina, se presenta una presentacion de un indice de felicidad constriudo a partir de 10 diferentes variables de 60 paises ")
st.subheader("Explicacion variables usadas:")
st.write("1 Inflacion:")
st.write("2 IDH:")
st.write("3 PIB per Capita:")
st.write("4 :")
st.write("5 Tasa de suicidios:")
st.write("6 Salario medio de los empleados:")
st.write("7 Desempleo total:")
st.write("8 Esperanza de vida:")
st.write("9 Poblacion mayor a los 65 años:")
st.write("10 Tasa de homicidios intencionales:")

st.subheader("Reduccion de dimensionalidad (Algoritmo usado PCA):")
st.subheader("Calificación Davies-Boluldin:")
st.subheader("Algoritmo de Clustering seleccionado Kmeans:")
st.subheader("Concluya sobre los clústers de manera descriptiva y gráfica.")
st.subheader("Análisis descriptivo y gráfico")
st.subheader("Conclusión General")
st.write("[Link al Notebook](https://colab.research.google.com/drive/115jkwsUACKRFmJAgcehE8PxzRzFha0Y1?usp=sharing)")

X = tabla.to_numpy()
scal = StandardScaler()
X_scal = scal.fit_transform(tabla)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scal)

kmeans = KMeans(n_clusters=3, random_state=777,algorithm='elkan').fit(X_scal)
pca_3['labels'] = kmeans.labels_
Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
labels = kmeans.labels_
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

st.plotly_chart(fig,use_container_widht=True)

