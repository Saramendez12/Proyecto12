!pip install wbgapi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wbgapi as wb
from sklearn.linear_model import LinearRegression

from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py

tabla=wb.data.DataFrame(["FP.CPI.TOTL.ZG","HD.HCI.OVRL","NY.GDP.PCAP.CD","SE.XPD.TOTL.GD.ZS","SH.STA.SUIC.P5",
                         "SL.EMP.WORK.ZS","SL.UEM.TOTL.ZS","SP.DYN.LE00.IN","SP.POP.65UP.TO.ZS","VC.IHR.PSRC.P5"],
                        ["ALB","DEU","ARG","ARM","AUS","AUT","BGD","BOL","BRA","BTN","CAN","CHN","COL","KOR","CRI",
                         "DNK","SLV","ARE","ESP","USA","EST","FJI","FIN","FRA","GRC","IND","IDN","ISL","IRL","ISR",
                         "JAM","JPN","LBN","MYS","MAR","MEX","PAK","PER","URY","POL","QAT","GBR","DOM","SRB","ZAF",
                         "CHE","THA","TZA","TUN","TUR","UKR","YEM","AGO","SYR","SGP","RUS","SDN","KAZ","PNG","NLD"],range(2017,2018))
tabla.rename(columns={'FP.CPI.TOTL.ZG':"Inflacion_A", 'HD.HCI.OVRL':"IDH", 
                      'NY.GDP.PCAP.CD':"PIB_perca",'SE.XPD.TOTL.GD.ZS':"GPEDU_PIB",
                      'SH.STA.SUIC.P5':"Suicidio_rate", 'SL.EMP.WORK.ZS':"Salario empleados", 
                      'SL.UEM.TOTL.ZS':"Desempleo_total", 'SP.DYN.LE00.IN':"Esperanza_vida",
                      'SP.POP.65UP.TO.ZS':"POB_+65", 'VC.IHR.PSRC.P5':"Homicidios_inte"},inplace=True)
tabla
