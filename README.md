## Olá! Essa página é dedicada ao Minicurso online de Sensoriamento com Deep Learning ministrado às quintas-feiras.

### Links úteis:
<a href="https://www.sentinel-hub.com/" target="_blank"><img src="https://img.shields.io/badge/-SentinelHub-%230077B5?style=for-the-badge&logo=lattes&logoColor=white" target="_blank"></a> 

<a href="https://www.sentinel-hub.com/explore/sentinelplayground/" target="_blank"><img src="https://img.shields.io/badge/-Sentinel Playground-%2300?style=for-the-badge&logo=lattes&logoColor=white" target="_blank"></a> 

<a href="https://eos.com/landviewer/" target="_blank"><img src="https://img.shields.io/badge/-EOS LandViewer-%23333?style=for-the-badge&logo=lattes&logoColor=white" target="_blank"></a> 

---
### Materiais para programação em Python:
<p>Think Python: <a href="https://greenteapress.com/wp/think-python/"> Link</p>

<p> The Python Tutorial: <a href="https://docs.python.org/3/tutorial/"> Link</p>
 
---
### Livros:
CAMPBELL, James B.; WYNNE, Randolph H. Introduction to remote sensing. Guilford Press, 2011.

MANICKAVASAGAN, Annamalai; JAYASURIYA, Hemantha (Ed.). Imaging with electromagnetic spectrum: applications in food and agriculture. Springer, 2014.

LILLESAND, Thomas; KIEFER, Ralph W.; CHIPMAN, Jonathan. Remote sensing and image interpretation. John Wiley & Sons, 2015.

---
### Bibliotecas / Pacotes utilizados (Aula 3):
```js
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from os import listdir
from os.path import isfile, join
from rasterio.enums import Resampling
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
```

### Bibliotecas / Pacotes utilizados (Aula 5):
```js
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
```
