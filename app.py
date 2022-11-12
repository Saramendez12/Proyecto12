import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.graph_objs as go


pca_3=pd.read_csv("./pca_3.csv")
tabla=pd.read_csv("./tabla.csv")

st.title("Proyecto 12")
st.write("En la presente pagina, se realiza el despliegue del proyecto que buscaba construir un dataset con variables extraidas de la API del Banco Mundial que puedan explicar la felicidad de un país a partir de 10 diferentes variables de 60 paises")
st.subheader("EXPLICACIÓN VARIABLES USADAS")
st.write("1 Inflación, precios al consumidor (% anual):Cambio porcentual anual en el costo promedio de adquirir una canasta de bienes y servicios")
st.write("2 Indice de Desarrollo Humano:Calcula las contribuciones de la salud y la educación a la productividad del trabajador. El puntaje del índice varía de 0 a 1")
st.write("3 PIB per capita:Es el producto interno bruto dividido por la población a mitad de año.")
st.write("4 Gasto Publico en Educación (% del PIB):El gasto del gobierno general en educación (corriente, capital y transferencias) se expresa como porcentaje del PIB.")
st.write("5 Tasa de suicidios:Es el número de muertes por suicidio en un año por cada 100.000 habitantes.")
st.write("6 Salario medio de los empleados:")
st.write("7 Desempleo total:Se refiere a la proporción de la fuerza laboral que no tiene trabajo pero que está disponible y buscando empleo.")
st.write("8 Esperanza de vida: Indica el número de años que viviría un recién nacido")
st.write("9 Poblacion mayor a los 65 años:")
st.write("10 Tasa de homicidios intencionales:Son estimaciones de homicidios ilegales infligidos")

st.subheader("Reducción de dimensionalidad (Algoritmo usado PCA): Este metodo se realiza a través de una descomposición de la varianza")
st.write("jolll")

Scene = dict(xaxis = dict(title  = 'PCA1'),yaxis = dict(title  = 'PCA2'),zaxis = dict(title  = 'PCA3'))
trace = go.Scatter3d(x=pca_3['PCA1'], y=pca_3['PCA2'], z=pca_3['PCA3'], mode='markers',marker=dict(color = 'green', size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()
st.plotly_chart(fig,use_container_widht=True)

st.subheader("Calificación Davies-Boluldin:")
st.write("aaaaaaa")

st.subheader("Algoritmo de Clustering seleccionado Kmeans:")
st.subheader("Concluya sobre los clústers de manera descriptiva y gráfica.")
fig_pickle12 = open('fig_proyecto12.pickle','rb')
figi = pkl.load(fig_pickle12)
st.plotly_chart(figi,use_container_widht=True)
st.subheader("Análisis descriptivo y gráfico")
st.subheader("Conclusión General")
st.write("[Link al Notebook](https://colab.research.google.com/drive/115jkwsUACKRFmJAgcehE8PxzRzFha0Y1?usp=sharing)")



