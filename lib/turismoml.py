# --------------------------------------
#
# Modelo geográfico para la ampliación de alojamientos turísticos
#
# --------------------------------------
import pandas as pd

# --------------------------------------
#
# Algunas configuraciones
#
# --------------------------------------
# D: cambiar develop por main
baseUrl = "https://raw.githubusercontent.com/malkab/turismo_geo_ml/develop/"

# --------------------------------------
#
# Carga de los datos de encuesta. Devuelve un DataFrame Pandas a partir de los
# datos leidos del CSV.
#
# --------------------------------------
def cargaEncuestas():
  # D: develop por main
  url = baseUrl + "datos/encuestas.csv"
  return pd.read_csv(url)
