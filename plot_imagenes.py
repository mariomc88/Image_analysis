import pydicom
import matplotlib.pyplot as plt
import numpy
def plot_imagenes(*imagen, names):

    """
    Función plot_images(imagen)

    Descripción:
        Realiza un plot de todas las imágenes que recibe como argumento de entrada

    Argumento de entrada:
        - imagen > array con una o varias imágenes que van a ser representadas

    """
    num_img = len(imagen) #Cantidad de imagenes introducidas
    plt.figure(figsize = (6*num_img, 2*num_img))
    for subplot in range(num_img):
        name = names[subplot] #Nombre de la imagen
        imagen = pydicom.dcmread(imagen[subplot],force=True) #Leemos la imagen dicom
        imagen = imagen[subplot].pixel_array #Nos quedamos con el array de la imagen
        imagen = imagen[subplot]/imagen[subplot].max()# Normalizamos la imagen entre 0 y 1.
        plt.subplot(100 + num_img * 10 + subplot + 1);plt.imshow(imagen[subplot], cmap = "gray");plt.title(name) #Cada imagen se añade a un subplot con su respectivo título
        subplot += 1
    plt.show()