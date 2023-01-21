#Introducimos la semilla
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import pydicom
import matplotlib
from skimage import measure, viewer

def get_points(image):
    matplotlib.use('TkAgg') # Para que se pueda abrir una nueva ventana
    #plt.ion()
    plt.title("Seleccione puntos con tecla izquierda, \n borre el último punto con tecla derecha,\n finalize con tecla central", fontsize=16) #Instrucciones a seguir por el usuario para la exitosa extracción #de los puntos
    plt.imshow(image, cmap = 'gray') #Imagen sobre la que vamos a extraer los puntos
    #plt.pause(0.001) 
    seed_coord = plt.ginput(-1) #No ponemos límite al número de puntos que se pueden seleccionar a partir de ginput
    plt.close()
    return seed_coord #Devuelve una lista con las coordenadas de los puntos, se debe tener en cuenta las coordenadas las devuleve a modo y, x

def RegionGrowingP2(image, umbr_inf,umbr_max,seed_coord,alpha=0.7, connectivity = 2):
    #seed_coord = get_points(image)
    binarized_image = np.zeros(np.shape(image)+ ((len(seed_coord),))) #Creamos un array 3D de dimensiones las de la imagen y el número de puntos seleccionados, donde almacenar la imagen binarizada para cada #punto de seed_coord
    labeled_image = binarized_image #Array 3D de las mismas dimensiones que binarized_image donde almacenar la imagen con la region de vecinos dentro del rango para cada punto de seed_coord
    i = 0 #Variable contador de iteraciones
    for y,x in seed_coord: #Obtenemos las coordenadas x e y de cada punto de seed_coord a lo largo del bucle
        x = round(x) #No es posible trabajar con valores float de pixeles por lo que redondeamos este valor para mantenernos lo más cerca posible de las coordenadas de la selección.
        y = round(y)
        intensity = image[x,y] #Intensidad del pixel para la coordenada i dentro de seed_coord
        binarized_image[:,:,i] = np.logical_and(image > intensity-umbr_inf, image < intensity+umbr_max) #En las dos primeras dimensiones de nuestro array 3D añadimos la imagen binaria donde los pixeles entre #que se encuentran a partir del valor de intensidad seleccionado dentro del rango especificado toman el valor 1 y los demás 0
        labeled_image[:,:,i] = measure.label(binarized_image[:,:,i], background=False, connectivity = connectivity) #Esta función de skimage crea parches con distintos valores (desde 1 hasta el número de parches total) donde #haya pixeles con vecindad a 8 u a 4 que estén en contacto, todo esto en los valores 1 de la imagen binaria
        label = labeled_image[:,:,i][x,y] #Obtenemos el valor de parche que se cooresponde con las coordenadas de nuestro punto
        labeled_image[:,:,i] = (labeled_image[:,:,i]==label) #Creamos una imagen binaria donde solo sean 1 los pixeles dentro del parche de vecinos al seleccionado
        i += 1
    segmented_img = np.sum(labeled_image, axis = 2) #Sumamos las imagenes a lo largo de la tercera dimension (numero de puntos) de modo que los parches extraidos para cada coordenada se superpongon.
    segmented_img = (segmented_img != 0)*alpha #Ya que donde se superpongan distintos parches los valores serán superiores a 1, volvemos a binarizar la imagen y multiplicamos por el valor de transparencia #(alpha) que asignamos a nuestra segmentacion
    final_img = image + segmented_img #Sumamos a la imagen original la imagen segmentada binarizada para que se pueda observar la superposición
        
    plt.figure(figsize = (18,6))
    plt.subplot(131);plt.imshow(image,cmap="gray");plt.title("Imagen Original")
    plt.subplot(132);plt.imshow(segmented_img,cmap="gray");plt.title("Segmentacion crec. regiones");
    plt.subplot(133);plt.imshow(final_img,cmap="gray");plt.title("Segmentacion superpuesta")
    plt.show()
    
    #return segmented_img, final_img #Nuestra función devuelve tanto la imagen con la segmentación como la imagen con la suma de esta y la imagen original
    




