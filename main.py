import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time

ruta_origen = "/home/siali/github/facsa/FACSA/pruebas/dataset_images_forzada_22_feb"
ruta_buena = "/home/siali/github/facsa/FACSA/pruebas/buenas"
ruta_mala = "/home/siali/github/facsa/FACSA/pruebas/malas"

archivos = os.listdir(ruta_origen)
# archivos = [archivo for archivo in archivos if ".png" in archivo]

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

def presentacion(imagen, grafica1=None, grafica2=None, v=None):
    if (v<=65):
        deteccion = "Muy oscuro"
    elif (v>=65 and v<90):
        deteccion = "Oscuro"
    elif (v>=90 and v<110):
        deteccion = "Buen estado"
    elif (v>110):
        deteccion = "Muy claro"
    else:
        deteccion = None
    
    labels = ['h', 's', 'v']
    
    # Crear figura y subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Agregar imagen
    ax1.imshow(imagen)
    ax1.set_title(f'Detección: {deteccion}')

    # Agregar gráficas si se proporcionan
    if grafica1 is not None:
        x1 = np.arange(len(grafica1[0]))
        ax2a = plt.subplot2grid((2, 2), (0, 1))
        for i in range(3):
            ax2a.plot(x1, grafica1[i], label=labels[i])
        ax2a.set_title(f'Histograma del agua (tuberia 1)\nPromedio de V: {v}')
        ax2a.legend()
        
    if grafica2 is not None:
        x2 = np.arange(len(grafica2[0]))
        ax2b = plt.subplot2grid((2, 2), (1, 1))
        for i in range(3):
            ax2b.plot(x2, grafica2[i], label=labels[i])
        ax2b.set_title('Histograma del fondo')
        ax2b.legend()

    # Obtener el objeto del administrador de figuras actual y cambiar al modo de pantalla completa
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()

    # Mostrar la figura
    plt.show()


t1 = time.time()
for archivo in archivos[:3]:
    src_file = os.path.join(ruta_origen, archivo)
    
    print(src_file)
    
    imagen = cv2.imread(src_file)
    copia = imagen.copy()
    
    # ROIs
    tub1 = [360, 425, 775, 925]
    tub2 = [500, 575, 1225, 1325]
    chorro = [500, 1000, 500, 1000]
    fon = [300 , 1800, 150, 300]
    tuberia1 = imagen[tub1[0]:tub1[1], tub1[2]:tub1[3]]
    tuberia2 = imagen[tub2[0]:tub2[1], tub2[2]:tub2[3]]
    chorro = imagen[chorro[0]:chorro[1], chorro[2]:chorro[3]]
    fondo = imagen[fon[0]:fon[1], fon[2]:fon[3]]

    # Dibujo de ROIs
    cv2.rectangle(copia, (tub1[2], tub1[0]), (tub1[3], tub1[1]), (0, 255, 0), 2)
    cv2.rectangle(copia, (tub2[2], tub2[0]), (tub2[3], tub2[1]), (0, 255, 0), 2)
    cv2.rectangle(copia, (fon[2], fon[0]), (fon[3], fon[1]), (0, 255, 0), 2)

    cv2.putText(copia, "Tuberia_1", (tub1[2], tub1[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
    cv2.putText(copia, "Tuberia_2", (tub2[2], tub2[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
    cv2.putText(copia, "Fondo", (fon[2], fon[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)

    
    h, s, v = histograma(tuberia1, color='hsv', rep=False)
    hf, sf, vf = histograma(fondo, color='hsv', rep=False)
    
    # Quitar los últimos picos
    v[-5:] = 0

    # Criterio 1: valor modal
    fmax = int(np.amax(v))
    v_modal = np.argmax(v)

    # Criterio 2: media ponderada del valor
    t = np.arange(len(v)).reshape(-1, 1)
    v_modal2 = int(round(float(np.average(t, weights=v, axis=0))))

    print(f"Valor más repetido = {v_modal2}. Repetido {fmax} veces.")
    # print(f"Valor más repetido = {v_modal2} (con media).")
    
    # if v_modal >= 160:
    #     cv2.imwrite(f"{ruta_buena}/{archivo}.jpg", copia)
    # else:
    #     cv2.imwrite(f"{ruta_mala}/{archivo}.jpg", copia)
        
    # if os.path.exists(src_file):
    #     if v_modal >= 160:
    #         dst_file = os.path.join(ruta_buena, archivo)
    #     else:
    #         dst_file = os.path.join(ruta_mala, archivo)
    #     shutil.move(src_file, dst_file)

    representar_imagen = False
    if representar_imagen:
        cv2.imshow("Imagen", copia)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    
    # Llamar a la función presentacion()
    grafica1 = [h, s, v]
    # grafica2 = [x2, y2]
    presentacion(copia, grafica1, v=v_modal2)
    
       
        
        
        
        
    
t2 = time.time()
print(f"Tiempo total = {t2-t1}")
    