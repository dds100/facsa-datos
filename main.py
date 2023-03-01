import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time

# En este script estaba intentando impolementar una función para distinguir si
# sale agua o no de las tuberías.

# Número de imágenes a copiar
n_imagenes = 500

# Limites de valor
limites = [70, 100, 130]

# Ruta origen
ruta_origen = "/home/siali/github/facsa/FACSA/Pruebas/dia_entero_con_agua"

# Rutas destino
ruta_muyoscuro = "/home/siali/github/facsa/FACSA/clasificacion_color/MuyOscuro"
ruta_oscuro = "/home/siali/github/facsa/FACSA/clasificacion_color/Oscuro"
ruta_buenestado = "/home/siali/github/facsa/FACSA/clasificacion_color/BuenEstado"
ruta_muyclaro = "/home/siali/github/facsa/FACSA/clasificacion_color/MuyClaro"

# Lista de archivos
archivos = os.listdir(ruta_origen)
archivos.sort()

# Funciones
def histograma(imagen, color='rgb', titulo='Histograma', rep=True):
    colores = ['r', 'g', 'b']
    if color == 'rgb':
        labels = colores
    elif color == 'hsv':
        labels = ['h', 's', 'v']
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    canales = cv2.split(imagen)
    frecuencias = [cv2.calcHist([i], [0], None, [256], [0, 256]) for i in canales]
    if rep:
        lineas = [plt.plot(frecuencias[j], color=colores[j], label=labels[j]) for j in range(len(frecuencias))]        
        plt.title(titulo)
        plt.xlim([0, 256])
        plt.legend()
        plt.show()
    return [k for k in frecuencias]

def calculo_v(array_v, criterio=0):
    # Quitar los últimos picos del final (ruido)
    array_v[-5:] = 0
    if criterio == 0:
        # Criterio 1: valor modal
        v_modal = np.argmax(array_v)
    if criterio == 1:
        # Criterio 2: media ponderada del valor
        t = np.arange(len(array_v)).reshape(-1, 1)
        v_modal = int(round(float(np.average(t, weights=array_v, axis=0))))
    return v_modal

def clasificacion(v, vchorro):
    if hay_agua(v, vchorro):
        if (v<=limites[0]):
            deteccion = 0
        elif (v>=limites[0] and v<limites[1]):
            deteccion = 1
        elif (v>=limites[1] and v<limites[2]):
            deteccion = 2
        elif (v>=limites[2]):
            deteccion = 3
    else:
        deteccion = 4
    return deteccion

def presentacion_cutre(imagen, titulo=""):
    fig, ax = plt.subplots()
    ax.imshow(imagen)
    title_style = {'family': 'serif', 'color': 'red', 'weight': 'bold', 'size': 64}
    ax.set_title('Título de la imagen', fontdict=title_style)
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    plt.show()

def presentacion(imagen, grafica1=None, grafica2=None, v=None, titulo_imagen="Imagen", titulo1="Gráfica 1", titulo2="Gráfica 2"):
    labels = ['h', 's', 'v']
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imagen)
    ax1.set_title(titulo_imagen)
    if grafica1 is not None:
        x1 = np.arange(len(grafica1[0]))
        ax2a = plt.subplot2grid((2, 2), (0, 1))
        for i in range(3):
            ax2a.plot(x1, grafica1[i], label=labels[i])
        ax2a.set_title(titulo1)
        ax2a.legend()  
    if grafica2 is not None:
        x2 = np.arange(len(grafica2[0]))
        ax2b = plt.subplot2grid((2, 2), (1, 1))
        for i in range(3):
            ax2b.plot(x2, grafica2[i], label=labels[i])
        ax2b.set_title(titulo2)
        ax2b.legend()
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    plt.show()
    
def guardar_imagen(archivo, imagen, v, deteccion, indice, rep=False):
    if type(deteccion) == str:
        deteccion = deteccion.replace(" ", "")
    nombre_archivo = f"{archivo.split('.')[0]}_{deteccion}_tub{indice}_v{v}.jpg"
    if (v<=limites[0]):
        cv2.imwrite(f"{ruta_muyoscuro}/{nombre_archivo}", imagen)
    elif (v>=limites[0] and v<limites[1]):
        cv2.imwrite(f"{ruta_oscuro}/{nombre_archivo}", imagen)
    elif (v>=limites[1] and v<limites[2]):
        cv2.imwrite(f"{ruta_buenestado}/{nombre_archivo}", imagen)
    elif (v>limites[2]):
        cv2.imwrite(f"{ruta_muyclaro}/{nombre_archivo}", imagen)
    else:
        return "Error"
    if rep:  
        print(archivo)

def mover_imagen():
    pass

def peor_tuberia(tupla_v):
    if min(tupla_v) < limites[1]:
        return min(tupla_v), tupla_v.index(max(tupla_v))+1
    else:
        return max(tupla_v), tupla_v.index(max(tupla_v))+1
    
def borrar_contenido(carpeta):
    files = os.listdir(carpeta)
    for file in files:
        os.remove(f"{carpeta}/{file}")
        
def hay_agua(vtub, vchorro, distancia_max=50):
    return vchorro in range(vtub-distancia_max, vtub+distancia_max)

def distancia_valor(v, v_ideal=115):
    return round(100*(1-abs(v_ideal-v)/v_ideal))

if __name__ == "__main__":
    
    # Para medir el tiempo total
    t1 = time.time()
    
    # Borro los archivos de las carpetas de pruebas
    borrar = False
    if borrar:
        borrar_contenido(ruta_buenestado)
        borrar_contenido(ruta_muyclaro)
        borrar_contenido(ruta_oscuro)
        borrar_contenido(ruta_muyoscuro)
    
    # Contador de imagenes
    cont = 0
    
    # Recorro los archivos de la carpeta origen
    for archivo in archivos:
        
        # Ruta de cada imagen
        src_file = os.path.join(ruta_origen, archivo)
            
        # Leer imagen
        imagen = cv2.imread(src_file)
        copia = imagen.copy()
        
        # ROIs
        tuberia1 = [360, 425, 775, 925]
        tuberia2 = [540, 590, 1225, 1325]
        chorro1 = [550, 600, 775, 875]
        chorro2 = [600, 725, 1100, 1250]
        fondo = [300 , 1800, 150, 300]
        imagen_tuberia1 = imagen[tuberia1[0]:tuberia1[1], tuberia1[2]:tuberia1[3]]
        imagen_tuberia2 = imagen[tuberia2[0]:tuberia2[1], tuberia2[2]:tuberia2[3]]
        imagen_chorro1 = imagen[chorro1[0]:chorro1[1], chorro1[2]:chorro1[3]]
        imagen_chorro2 = imagen[chorro2[0]:chorro2[1], chorro2[2]:chorro2[3]]
        imagen_fondo = imagen[fondo[0]:fondo[1], fondo[2]:fondo[3]]

        # Cálculo de histogramas
        h, s, v = histograma(imagen, color='hsv', rep=False)
        h1, s1, v1 = histograma(imagen_tuberia1, color='hsv', rep=False)
        h2, s2, v2 = histograma(imagen_tuberia2, color='hsv', rep=False)
        hch1, sch1, vch1 = histograma(imagen_chorro1, color='hsv', rep=False)
        hch2, sch2, vch2 = histograma(imagen_chorro2, color='hsv', rep=False)
        hf, sf, vf = histograma(imagen_fondo, color='hsv', rep=False)
        
        # Cálculo de v
        v_modal_tub1 = calculo_v(v1, criterio=1)
        v_modal_tub2 = calculo_v(v2, criterio=1)
        v_modal_ch1 = calculo_v(vch1, criterio=1)
        v_modal_ch2 = calculo_v(vch2, criterio=1)
        tupla_v = (v_modal_tub1, v_modal_tub2)
        
        # # Peor valor de las dos tuberías
        # v_modal_tub, indice = peor_tuberia(tupla_v)
        
        # Estados posibles
        estados = ["Muy oscuro", "Oscuro", "Buen estado", "Muy claro", "Sin agua"]
        
        # Comprobación de agua
        agua1 = hay_agua(v_modal_tub1, v_modal_ch1, distancia_max=50)
        agua2 = hay_agua(v_modal_tub2, v_modal_ch2, distancia_max=50)
        
        # Clasificación de la imagen según v
        deteccion1 = clasificacion(v_modal_tub1, v_modal_ch1)
        deteccion2 = clasificacion(v_modal_tub2, v_modal_ch2)

        # Colores
        colores_posibles = {"Rojo": (255, 0, 0), "Naranja": (255, 128, 0),"Amarillo": (255, 255, 0), "Verde": (0, 255, 0), "Azul": (0, 255, 255)}
        colores = ["Rojo", "Naranja", "Verde", "Amarillo", "Azul"]
        
        # Dibujo de ROIs
        cv2.rectangle(copia, (tuberia1[2], tuberia1[0]), (tuberia1[3], tuberia1[1]), colores_posibles[colores[deteccion1]], 2)
        cv2.rectangle(copia, (tuberia2[2], tuberia2[0]), (tuberia2[3], tuberia2[1]), colores_posibles[colores[deteccion2]], 2)
        # cv2.rectangle(copia, (chorro1[2], chorro1[0]), (chorro1[3], chorro1[1]), (0, 255, 0), 2)
        # cv2.rectangle(copia, (chorro2[2], chorro2[0]), (chorro2[3], chorro2[1]), (0, 255, 0), 2)
        # cv2.rectangle(copia, (fondo[2], fondo[0]), (fondo[3], fondo[1]), (255, 0, 255), 2)

        cv2.putText(copia, f"Tuberia 1: {distancia_valor(v_modal_tub1)}% / {estados[deteccion1]}", (tuberia1[2], tuberia1[0]-10), cv2.FONT_HERSHEY_COMPLEX, .8, colores_posibles[colores[deteccion1]], 2)
        cv2.putText(copia, f"Tuberia 2: {distancia_valor(v_modal_tub2)}% / {estados[deteccion2]}", (tuberia2[2], tuberia2[0]-10), cv2.FONT_HERSHEY_COMPLEX, .8, colores_posibles[colores[deteccion2]], 2)
        # cv2.putText(copia, "Chorro_1", (chorro1[2], chorro1[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
        # cv2.putText(copia, "Chorro_2", (chorro2[2], chorro2[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
        # cv2.putText(copia, "Fondo", (fondo[2], fondo[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 0, 255), 2)

        # # Guardado de las imágenes clasificadas
        # guardar_imagen(archivo, imagen, v_modal, deteccion, indice)
        
        # Imshow de la imagen
        representar_imagen = False
        if representar_imagen:
            cv2.imshow("Imagen", copia)
            # cv2.imshow("Real", imagen_chorro1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Imagen con gráficas de histogramas
        presentar = False
        if presentar:
            grafica1 = [h1, s1, v1]
            grafica2 = [hch1, sch1, vch1]
            presentacion(copia, grafica1, grafica2, titulo_imagen=f"Agua en tuberias -> 1: {estados[deteccion1]} 2: {estados[deteccion2]}", titulo1=f"Histograma tuberia 1, v = {v_modal_tub1}", titulo2=f"Histograma chorro 1, v = {v_modal_ch1}")
        
        # Imagen a pelo con detecciones
        presentar_cutre = True
        if presentar_cutre:
            presentacion_cutre(copia)#, titulo=f"Agua en tuberias    -->     1: {estados[deteccion1]}    2: {estados[deteccion2]}")
        
        # Contador de imagenes
        cont += 1
        
        # # Mostrar el proceso en consola
        # print(f"{cont}/{len(archivos)} - {archivo} - {agua1}")

    # Para medir el tiempo total
    t2 = time.time()
    print(f"Tiempo total = {round(t2-t1, 2)} segundos.")
