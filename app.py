import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Proyecto 12")
st.write(" En la presente pagina, se presenta una presentacion de un indice de felicidad constriudo a partir de 10 diferentes variables de 60 paises ")
st.subheader("Explicacion variables usadas: ")
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

st.subheader("Reduccion de dimencionalidad (PCA): ")
st.subheader("Cluster y Kmeans:")

st.plotly_chart(fig,use_container_widht=True)

