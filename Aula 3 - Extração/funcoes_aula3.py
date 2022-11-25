# -*- coding: utf-8 -*-
"""
TUTORIAL

Carregando imagens do Sentinel-2 com Python
"""
import numpy as np
import rasterio
from os import listdir
from os.path import isfile, join
from rasterio.enums import Resampling
from rasterio.mask import mask

def create_dataset(path_input,my_map):
    # Obtendo nomes de arquivos de imagem
    img_nomes = [f for f in listdir(path_input) if isfile(join(path_input, f))]
 
    #Obtendo Bandas
    Bandas = get_bands(path_input, img_nomes, my_map)
 
    #Índices de vegetação
    VI = get_VIs(Bandas)
    
    #Combinações
    Combinacoes = get_combination(Bandas)
    
    #Concatenando informações
    data = np.concatenate((Bandas,VI),axis=2)
    data = np.concatenate((data,Combinacoes),axis=2)
    return data

#Recortando imagens
def cut_images(path_input,img_nome, my_map):
    full_path = join(path_input,img_nome)
    with rasterio.open(full_path) as src:       
        out_image, _ = mask(src, [my_map], crop=True)
        #out_meta = src.meta.copy()
        out_image.squeeze()
    return out_image

#Reamostrando Imagem
def interpolation_method(path_input,img_nome, my_map, factor):  
    full_path = join(path_input,img_nome)
    with rasterio.open(full_path) as src:
        out_meta = src.meta
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * factor),
                int(src.width * factor)
            ),
            resampling=Resampling.bilinear #reamostragem bilinear
        )
        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
    
        out_meta.update({"driver": "GTiff",
                         "height": data.shape[1],
                         "width": data.shape[2],
                         "transform": transform})
    data = data.squeeze(0)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(data, 1)
        del data
        with memfile.open() as dataset:
            out_image, _ = mask(dataset, [my_map], crop=True)
            return out_image.squeeze()

def get_bands(path_input, img_nomes, my_map):
    k = 0
    for nomes in img_nomes:
        if ('.TIF' in nomes):
            if ('B02' in nomes) or ('B03' in nomes) or ('B04' in nomes) or ('B08.' in nomes):
                b_aux = cut_images(path_input, nomes, my_map)
            else:
                if ('B01' in nomes) or ('B09' in nomes):
                    b_aux = interpolation_method(path_input, nomes, my_map,6)   
                else:
                    b_aux = interpolation_method(path_input, nomes, my_map,2)    
            if k == 0:
                Bandas = np.zeros((b_aux.shape[0],b_aux.shape[1],12))  
            Bandas[:,:,k] = b_aux
            k = k+1
            print(nomes)
    return Bandas

def get_combination(Bandas):
    #Combinações RGB
    #Color Infrared (B8, B4, B3)
    l,c,_ = Bandas.shape
    Combina = np.zeros((l,c,3)) 
    aux = np.zeros((l,c,3))
    aux[:,:,0] = Bandas[:,:,7]
    aux[:,:,1] = Bandas[:,:,3]
    aux[:,:,2] = Bandas[:,:,2]
    Combina[:,:,0] = rgb2gray(aux)
    #plt.imshow(Color_Infrared.astype('uint8'))

    #Short-Wave Infrared (B12, B8A, B4)
    aux[:,:,0] = Bandas[:,:,11] 
    aux[:,:,1] = Bandas[:,:,8]
    aux[:,:,2] = Bandas[:,:,3]
    Combina[:,:,1] = rgb2gray(aux)
    
    #Agriculture (B11, B8, B2)
    aux[:,:,0] = Bandas[:,:,10] 
    aux[:,:,1] = Bandas[:,:,7]
    aux[:,:,2] = Bandas[:,:,1]
    Combina[:,:,2] = rgb2gray(aux)
    return Combina

def get_pixels(path_input, my_map):
    #Obtendo dados
    aux = create_dataset(path_input,my_map)
    
    #Obtendo matriz de pixeis
    l,c,_ = aux.shape
    alvo = np.zeros((l*c,19))
    k = 0
    for i in range(l):
        for j in range(c):
            alvo[k] = aux[i,j,:]
            k = k+1
    # remove rows having all zeroes
    alvo = alvo[~np.all(alvo == 0, axis=1)]
    return alvo

def get_VIs(Bandas):
    l,c,_ = Bandas.shape
    L=0.5
    VI = np.zeros((l,c,4))
    VI[:,:,0] = np.divide((Bandas[:,:,7] - Bandas[:,:,3]),(Bandas[:,:,7] + Bandas[:,:,3]))
    VI[:,:,1] = np.divide((Bandas[:,:,7] - Bandas[:,:,2]),(Bandas[:,:,7] + Bandas[:,:,2]))
    VI[:,:,2] = np.divide((Bandas[:,:,7] - Bandas[:,:,4]),(Bandas[:,:,7] + Bandas[:,:,4]))
    VI[:,:,3] = np.divide((Bandas[:,:,7] - Bandas[:,:,3]),(Bandas[:,:,7] + Bandas[:,:,3] + L)) * (1.0 + L)
    VI[np.isnan(VI)] = 0
    return VI
        
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.1140,0.5870,0.2989])

def select_random_pixel(array_pixel, num_rows):
    #Selecionando amostras de pixeis aleatoriamente sem reposição
    seed = 42
    A = np.copy(array_pixel)
    np.random.seed(seed)
    np.random.shuffle(A)
    array_8000 = A[:num_rows, :]
    return array_8000