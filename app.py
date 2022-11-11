import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

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


st.plotly_chart(fig,use_container_widht=True)

