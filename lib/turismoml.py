# --------------------------------------
#
# Modelo geográfico para la ampliación de alojamientos turísticos
#
# --------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kmodes.kprototypes import KPrototypes
import seaborn as sns
from mapboxgl.utils import create_color_stops
from mapboxgl.viz import MapViz, CircleViz, ChoroplethViz
import json


# --------------------------------------
#
# Algunas configuraciones
#
# --------------------------------------
# D: cambiar
# baseUrl = "https://raw.githubusercontent.com/malkab/turismo_geo_ml/main/"

# Configuración Seaborn
sns.set(rc = {'figure.figsize':(20,10)})

# Token Mapbox
token = "pk.eyJ1IjoibWFsa2FiIiwiYSI6ImNreDV0bTQ3ZjFhZXAyb28xMDJ3ZWpnczkifQ.6PXK4dwY4GurJuIaaZ5YJg"
mapHeight = "600px"
defaultCenter = (-6, 37.385)
defaultZoom = 11


# --------------------------------------
#
# Carga de los datos de encuesta. Devuelve un DataFrame Pandas a partir de los
# datos leidos del CSV.
#
# --------------------------------------
def cargaEncuestas(datos):

  # Cargamos datos desde la URL
  datos = pd.read_csv(datos)

  # Corregimos los Not a Number (NaN) en la columna "estudios"
  datos["estudios"] = datos["estudios"].fillna(False)

  return datos

# --------------------------------------
#
# Visualiza un histograma a partir de una variable continua
#
# --------------------------------------
def histograma(datos, range, bins=25, type="step"):

  plt.figure()
  plt.hist(datos, bins=bins, histtype=type, range=range)
  plt.show()

# --------------------------------------
#
# Visualiza un histograma a partir de una variable discretas
#
# --------------------------------------
def histogramaDiscretas(datos):

  datos.value_counts().plot(kind="bar")

# --------------------------------------
#
# Generación de clusters K-Modes
#
# --------------------------------------
def kprototypes(datos, clusters):

  km = KPrototypes(n_jobs=-1, n_clusters=clusters, init="Huang", n_init=150,
    random_state=0, verbose=1)
  clustersId = km.fit_predict(datos, categorical=[ 1,2,3,4,5,6,7,8 ])
  unique, counts = np.unique(clustersId, return_counts=True)

  clusters = []

  # Describir el resultado de los clusters
  for i in km.cluster_centroids_:
    clusters.append({
      "Edad media": i[0],
      "Acompañantes": i[1],
      "Tipo alojamiento": i[2],
      "Sol y playa": i[3],
      "Naturaleza / rural": i[4],
      "Cultural": i[5],
      "Deporte": i[6],
      "Gastronomia": i[7],
      "Estudios": i[8]
    })


  return clusters, clustersId, counts


# --------------------------------------
#
# Gráfica del codo para comprobar el número óptimo de clústeres
#
# --------------------------------------
def analisisCodo(datos, numClusters):

  cost = []
  K = range(numClusters[0], numClusters[1])

  for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Huang", n_init = 5, verbose=0)
    kmode.fit_predict(datos)
    cost.append(kmode.cost_)

  plt.plot(K, cost, 'bx-')
  plt.xlabel('Nº clusters')
  plt.ylabel('Coste')
  plt.title('Método del codo para estimar el número óptimo de clusters')
  plt.show()


# --------------------------------------
#
# Mostrar mapa de puntos,
#
# --------------------------------------
def mapaPuntos(data, e):

  # Load data from sample csv
  data_url = 'https://raw.githubusercontent.com/mapbox/mapboxgl-jupyter/master/examples/data/points.csv'
  df = pd.read_csv(data_url).round(3)

  # Generate data breaks using numpy quantiles and color stops from colorBrewer
  measure = 'Avg Medicare Payments'
  color_breaks = [round(df[measure].quantile(q=x*0.1), 2) for x in range(1,9)]
  color_stops = create_color_stops(color_breaks, colors='YlGnBu')

  data = json.loads(df.to_json(orient='records'))
  v = CircleViz(baseUrl + data,
                access_token=token,
                vector_url='mapbox://rsbaumann.2pgmr66a',
                vector_layer_name='healthcare-points-2yaw54',
                vector_join_property='Provider Id',
                data_join_property='Provider Id',
                color_property=measure,
                color_stops=color_stops,
                radius=4.5,
                stroke_color='black',
                stroke_width=0.2,
                center=(-95, 40),
                zoom=3,
                below_layer='waterway-label',
                legend_text_numeric_precision=0)
  v.show()


# --------------------------------------
#
# Mostrar mapa de polígonos interpolado.
#
# --------------------------------------
def mapaPoligonosInterpolado(datos, propiedadColor=None, valoresCorte=[],
  centro=defaultCenter, zoom=defaultZoom, opacidad=1,
  capaBase="waterway-label", funcionColor="interpolate"):

  if valoresCorte:
    colorStops = create_color_stops(valoresCorte, colors="RdYlBu")
  else:
    colorStops = None

  viz = ChoroplethViz(
    datos,
    access_token=token,
    color_property=propiedadColor,
    color_stops=colorStops,
    color_function_type=funcionColor,
    opacity=opacidad,
    center=centro,
    zoom=zoom,
    height=mapHeight,
    below_layer=capaBase)

  viz.show()


# --------------------------------------
#
# Mostrar mapa de polígonos discreto.
#
# --------------------------------------
def mapaPoligonosDiscreto(datos, propiedad=None, categorias=None,
  centro=defaultCenter, zoom=defaultZoom, opacidad=1,
  capaBase="waterway-label", colorDefecto="rgb(34,12,120"):

  viz = ChoroplethViz(
    datos,
    access_token=token,
    color_property=propiedad,
    color_stops=[],
    color_function_type="match",
    color_default=colorDefecto,
    opacity=opacidad,
    center=centro,
    zoom=zoom,
    height=mapHeight,
    below_layer=capaBase)

  viz.show()


# --------------------------------------
#
# Mostrar mapa de polígonos sencillo.
#
# --------------------------------------
def mapaPoligonosSimple(datos, nombreObjeto,
  centro=defaultCenter, zoom=defaultZoom, opacidad=1,
  capaBase="waterway-label", colorDefecto="rgb(111,255,102)"):

  viz = ChoroplethViz(
    datos,
    access_token=token,
    color_property="Leyenda",
    color_stops=[[ nombreObjeto, colorDefecto ]],
    color_function_type="match",
    color_default=colorDefecto,
    opacity=opacidad,
    center=centro,
    zoom=zoom,
    height=mapHeight,
    below_layer=capaBase)

  viz.show()


# --------------------------------------
#
# Mostrar mapa de puntos.
#
# --------------------------------------
def mapaPuntosInterpolado(datos, propiedadColor=None, valoresCorte=[],
  centro=defaultCenter, zoom=defaultZoom, opacidad=1,
  capaBase="waterway-label", colorBorde="black", radio=2, grosorBorde=0.2):

  if valoresCorte:
    colorStops = create_color_stops(valoresCorte, colors="YlGnBu")
  else:
    colorStops = None

  viz = CircleViz(
    datos,
    access_token=token,
    height=mapHeight,
    color_property=propiedadColor,
    color_stops=colorStops,
    radius=radio,
    stroke_color=colorBorde,
    stroke_width=grosorBorde,
    opacity=opacidad,
    center=centro,
    zoom=zoom,
    below_layer=capaBase)

  viz.show()
