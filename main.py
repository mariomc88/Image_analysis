# Importación de módulos
from skimage import measure, viewer, segmentation, filters, feature
from skimage.segmentation import active_contour
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
#from imimposemin import imimposemin
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pydicom


def plot_images(imagen = []):

    """
    Función plot_images(imagen)

    Descripción:
        Realiza un plot de todas las imágenes que recibe como argumento de entrada

    Argumento de entrada:
        - imagen > array con una o varias imágenes que van a ser representadas

    """
    num_img = len(imagen) #Cantidad de imagenes introducidas
    plt.figure(figsize = (6*num_img, 2*num_img))
    for subplot in range(len(imagen)):
        name = imagen[subplot] #Nombre de la imagen
        imagen[subplot] = pydicom.dcmread(imagen[subplot],force=True) #Leemos la imagen dicom
        imagen[subplot] = imagen[subplot].pixel_array #Nos quedamos con el array de la imagen
        imagen[subplot] = imagen[subplot]/imagen[subplot].max()# Normalizamos la imagen entre 0 y 1.
        plt.subplot(100 + num_img * 10 + subplot + 1);plt.imshow(imagen[subplot], cmap = "gray");plt.title("Imagen "+ name) #Cada imagen se añade a un subplot con su respectivo título
        subplot += 1
    plt.show()


def get_points(image):

    """
    Función get_points(image)

    Descripción:
        Extrae la imagen del archivo .dicom y permite seleccionar semillas utilizando .ginput()

    Argumentos de entrada:
        - image > nombre del archivo .dicom que contiene la imagen médica

    Resultado:
        - seed_coord > lista con las coordenadas de las semillas
        - img > imagen medica (array de píxeles)
    """

    ds = pydicom.dcmread(image,force=True) # cargamos el .dicom con pydicom
    img = ds.pixel_array # nos quedamos con la imagen médica
    img = img/img.max() # normalizamos la imagen
    matplotlib.use('TkAgg') # configuración de marplotlib para que se abra una nueva ventana
    plt.title("Seleccione puntos con tecla izquierda, \n borre el último punto con tecla derecha,\n finalize con tecla central", fontsize=16) # mostrar instrucciones para el usuario
    plt.imshow(img, cmap = 'gray') # mostrar imagen sobre la que se van a extraer los puntos
    seed_coord = plt.ginput(-1) # No ponemos límite al número de puntos que se pueden seleccionar con ginput
    plt.close()


    return seed_coord, img # Devuelve una lista con las coordenadas de los puntos, se debe tener en cuenta las coordenadas las devuleve a modo y, x (row, col)

def filtro_anisotropico(img, treshold = 0.3, size_filter = 3, iter = 3):

    for i in range(0,iter):
        if i != 0:
            img = mean_filtered
        sobel_filtered = filters.sobel(img)
        mean_filter = np.ones((size_filter,size_filter))/np.power(size_filter,2)
        mean_filtered = convolve(img, mean_filter,mode='reflect')
        pixeles_sobel = np.argwhere(sobel_filtered > treshold)
        mean_filtered[pixeles_sobel] = img[pixeles_sobel] #No hace falta el bucle for con esta linea basta.
    return mean_filtered

def RegionGrowingP2(image, umbr_inf, umbr_max, seed_coord, alpha=0.7, connectivity=2):

    """
    Funcionamiento del método de crecimiento de regiones:
        1. Selección de la(s) semilla(s) mediante la función ginput
        2. Binarización de la(s) imagen(es) en función del umbral superior e inferior (para cada punto se añade una imagen con la binarización a partir de la intensidad en ese punto)
        3. Marcación de parches de píxeles vecinos dentro de cada imagen binaria
        4. Almacenamos el parche en el que está incluida nuestra semilla en cada imagen.
        5. Superponemos todos los parches obtenidos a partir de todas las semillas y binarizamos el resultado a modo que todos los valores iguales o superiores a 1 pasen a ser 1, esto
        representará la imagen de la segmentación
        6. Superponemos la imagen de la segmentación a la imagen original para comprobar visualmente la bondad de la segmentación asignándole el valor de transparencia deseado.
        7. Finalmente se muestran por pantalla la imagen original, de la segmentación y la superposición de la segmentación sobre la original.

    Función RegionGrowingP2(image, umbr_inf, umbr_max, seed_coord, alpha=0.7, connectivity=2)

    Descripción:
        Realiza una segmentación por crecimiento de regiones y hace un plot de los resultados obtenidos (imagen original, región segmentada, segmentación superpuesta)

    Argumentos de entrada:
        - image > imagen sobre la que se va a realizar la segmentación
        - umbr_inf > rango inferior de valores de gris que se va a tener en cuenta para segmentar
        - umbr_max > rango superior de valores de gris que se va a tener en cuenta para segmentar
        - seed_coord > array con las coordenadas de las semillas (salida de la función 'get_points')
        - alpha (=0.7) > grado de transparencia de la capa superpuesta utilizado para el plot
        - connectivity (=2) > número de píxeles vecinos considerados para la conectividad (1 si es vecindad a 4 y 2 si es vecindad a 8)

    """

    binarized_image = np.zeros(np.shape(image)+ ((len(seed_coord),))) # Creamos un array 3D de dimensiones las de la imagen y el número de puntos seleccionados, donde almacenar la imagen binarizada para cada #punto de seed_coord
    labeled_image = binarized_image # Array 3D de las mismas dimensiones que binarized_image donde almacenar la imagen con la region de vecinos dentro del rango para cada punto de seed_coord
    i = 0 # Variable contador de iteraciones
    for y,x in seed_coord: # Obtenemos las coordenadas x e y de cada punto de seed_coord a lo largo del bucle
        x = int(round(x)) # No es posible trabajar con valores float de pixeles por lo que redondeamos este valor para mantenernos lo más cerca posible de las coordenadas de la selección.
        y = int(round(y))
        intensity = image[x,y] # Intensidad del pixel para la coordenada i dentro de seed_coord
        binarized_image[:,:,i] = np.logical_and(image > intensity-umbr_inf, image < intensity+umbr_max) # En las dos primeras dimensiones de nuestro array 3D añadimos la imagen binaria donde los pixeles entre #que se encuentran a partir del valor de intensidad seleccionado dentro del rango especificado toman el valor 1 y los demás 0
        labeled_image[:,:,i] = measure.label(binarized_image[:,:,i], background=False, connectivity = connectivity) # Esta función de skimage crea parches con distintos valores (desde 1 hasta el número de parches total) donde #haya pixeles con vecindad a 8 u a 4 que estén en contacto, todo esto en los valores 1 de la imagen binaria
        label = labeled_image[:,:,i][x,y] # Obtenemos el valor de parche que se cooresponde con las coordenadas de nuestro punto
        labeled_image[:,:,i] = (labeled_image[:,:,i]==label) # Creamos una imagen binaria donde solo sean 1 los pixeles dentro del parche de vecinos al seleccionado
        i += 1
    segmented_img = np.sum(labeled_image, axis = 2) # Sumamos las imagenes a lo largo de la tercera dimension (numero de puntos) de modo que los parches extraidos para cada coordenada se superpongon.
    segmented_img = (segmented_img != 0)*alpha # Ya que donde se superpongan distintos parches los valores serán superiores a 1, volvemos a binarizar la imagen y multiplicamos por el valor de transparencia #(alpha) que asignamos a nuestra segmentacion
    overlapped_img = image + segmented_img # Sumamos a la imagen original la imagen segmentada binarizada para que se pueda observar la superposición
    plt.figure(figsize = (18,6)) #Creamos la figura donde mostrar las imagenes
    plt.subplot(131);plt.imshow(image,cmap="gray");plt.title("Imagen Original")#Mostramos cada una de las imagenes en estas líneas
    plt.subplot(132);plt.imshow(segmented_img,cmap="gray");plt.title("Segmentacion crec. regiones");
    plt.subplot(133);plt.imshow(overlapped_img,cmap="gray");plt.title("Segmentacion superpuesta")
    plt.show()



def WatershedExerciseP2(image, seed_coord, suavizado=False, iters=3):

    """
    Funcionamiento del método de Watershed:
        1. Suavizado previo (o no) de la imagen utilizando filtro anisotrópico con el número de iteraciones correspondiente
        2. Se genera una imagen binarizada con las semillas (1 las semillas y 0 el resto)
        3. Aplicamos un filtrado de Sobel para quedarnos con imagen de relieves
        4. Transformamos los grises de la imagen con la función 'imimposemin' (a través de imagen binarizada y sobel)
        5. Aplicamos Watershed a la imagen transformada del paso anterior
        6. Plot de resultados

    Función WatershedExerciseP2(image, seed_coord, suavizado=False, iters=3)

    Descripción:
        Realiza una segmentación de Watershed y hace un plot de los resultados (sin y con semillas)

    Argumentos de entrada:
        - image > imagen que se va a utilizar para la segmentación
        - seed_coord > array con las coordenadas de las semillas
        - suavizado (=False) > indica si se va a aplicar un suavizado previo de la imagene
        - iters (=3) > número de iteraciones del filtro anisotrópico (en el caso de que se aplique)

    """

    if suavizado:
        image = filtro_anisotropico(image, treshold = 0.3, size_filter = 3, iter = iters) # filtrado anisotrópico

    binary_img = np.zeros(image.shape) # generamos matriz de 0's

    for coord in seed_coord:
        r, c = int(coord[1]), int(coord[0])
        binary_img[r,c] = 1 # damos valor 1 a las semillas introducidas

    sobel_img = filters.sobel(image) # aplicamos filtro sobel
    waters_img = segmentation.watershed(sobel_img)
    new_image = imimposemin(sobel_img, binary_img) # transformamos los grises con 'imimposemim' que recibe como entrada la imagen binaria y la de sobel
    waters_img2 = segmentation.watershed(new_image) # aplicamos watershed
    print('Número de semillas seleccionadas:',len(seed_coord))

    fig = plt.figure(figsize=(10, 8)) # plot del resultado
    ax1 = fig.add_subplot(1,2,1); ax1.imshow(waters_img,cmap='gray'); ax1.set_title('Imagen sin semilla')
    ax2 = fig.add_subplot(1,2,2); ax2.imshow(waters_img2,cmap='gray'); ax2.set_title('Imagen con semilla')


def snakes(img, seed_coord, radio,mode = "circulo", values_elipse = [], alpha = 0.015, beta=0.01, gamma=0.001, convergence = 0.8, w_edge = 5, coordinates = "rc"):

    '''
    Funcionamiento del algoritmo snakes:
        1. Selección del centro de la región, en este caso círculo de manera predeterminada(o elipse si se cambia la variable mode)
        2. Selección del radio del círculo (o radio mayor y menor de la elipse)
        3. Aplicación de snakes en la región seleccionada
        4.Plot de los resultados

    Función snakes(img, seed_coord, radio, mode = "circulo", values_elipse = [], alpha = 0.015, beta=0.01, gamma=0.001, convergence = 0.8, w_edge = 5, coordinates = "rc")

    Descripción:
        Aplicación del algoritmo de snakes para el seguimietnto de contornos

    Argumentos de entrada:
        - img > imagen que se va a utilizar para la segmentación
        - seed_coord > coordenada del centro del círculo o elipse
        - radio > radio del círculo o radio de inicio tanto para radio mayor como radio menor de la elipse (ajustados en values_elipse)
        - mode (="circulo") > forma geométrica que determina el área donde se aplica el algoritmo
        - values_elipse (=[]) > **********PREGUNTAAAAAAAARRRR A MARIO*******valores por los que se multiplica el radio inicial en la elipse. Se introducirá de la forma [radio]
        - alpha (=0.015) > parámetro que controla la longitud del snake
        - beta (=0.01) > parámetro que controla el suavizado del snake
        - gamma (=0.001) > parámetro que controla el stepping en snake
        - convergence (=0.8) > criterio de convergencia
        - w_edge (=5) > parámetro que controla la atracción a los bordes
        - coordinates (="rc") > utiliza como referencia coordenadas polares

    '''

    seed_coord = np.asarray(seed_coord[0])

    # creeación de la zona de trabajo
    s = np.linspace(0, 2*np.pi, 400)
    if mode == "circulo":
        r =  radio*np.sin(s) + seed_coord[1]
        c =  radio*np.cos(s) + seed_coord[0]
    elif mode == "elipse":
        r =  values_elipse[0]*radio*np.sin(s) + seed_coord[1]
        c =  values_elipse[1]*radio*np.cos(s) + seed_coord[0]
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=0.01, gamma=0.001,convergence = 0.8,w_edge = 5, coordinates = "rc")

    # representación gráfica
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # área de trabajo en rojo
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3) # snake en azul
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()
