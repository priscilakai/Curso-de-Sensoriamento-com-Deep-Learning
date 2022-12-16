# Abrindo uma imagem .TIF com Spyder

# Importação de Bibliotecas/Pacotes
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from os import listdir
from os.path import isfile, join
from rasterio.enums import Resampling
from rasterio.mask import mask

# Lendo KML files com GeoPandas
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
my_map = gpd.read_file('caminho_do_arquivo.kml', driver='KML')

# Convertendo EPSG para o mesmo da imagem do Sentinel-2
my_map = my_map.to_crs(epsg=32722)
    
# Pasta contendo images
path_input = 'caminho_contendo_as_imagens'

# Obtendo nomes de arquivos de imagem
img_nomes = [f for f in listdir(path_input) if isfile(join(path_input, f))]
full_path = join(path_input,img_nomes[3])

# Carregando dados de imagem em uma variável
arq = rasterio.open(full_path)
Banda4 = arq.read()
