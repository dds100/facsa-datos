import csv
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
from tqdm import tqdm
# from twilio.rest import Client

# # Credenciales Twilio para SMS
# # Set environment variables for your credentials
# account_sid = "AC02cefd407ce832492d7d9e4418366db0"
# auth_token = "273740116cc165550a194ce4ea91da4f"
# client = Client(account_sid, auth_token)

# Número de imágenes a copiar
n_imagenes = 500

# Limites de valor
valor_ideal = 120
limites = [70, 100, 130]

# Ruta origen
ruta_origen = "/home/siali/github/facsa/FACSA/Pruebas/Red/origen"
# ruta_origen = "/home/siali/github/facsa/FACSA/Pruebas/Red/origen_test"

# Rutas destino
ruta_muyoscuro = "/home/siali/github/facsa/FACSA/Pruebas/clasificacion_color/MuyOscuro"
ruta_oscuro = "/home/siali/github/facsa/FACSA/Pruebas/clasificacion_color/Oscuro"
ruta_buenestado = "/home/siali/github/facsa/FACSA/Pruebas/clasificacion_color/BuenEstado"
ruta_muyclaro = "/home/siali/github/facsa/FACSA/Pruebas/clasificacion_color/MuyClaro"
ruta_sinagua = "/home/siali/github/facsa/FACSA/Pruebas/clasificacion_color/SinAgua"
ruta_presentacion = "/home/siali/github/facsa/FACSA/Pruebas/presentacion"
ruta_etiquetado = "/home/siali/github/facsa/FACSA/Pruebas/Red/etiquetado"
ruta_etiquetado2 = "/home/siali/github/facsa/FACSA/Pruebas/Red/etiquetado_agua"

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
    # array_v[-3:] = 0
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
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imagen)
    ax1.set_title(titulo_imagen)
    if grafica1 is not None:
        x1 = np.arange(len(grafica1[0]))
        ax2a = plt.subplot2grid((2, 2), (0, 1))
        for i in range(1,3):
            ax2a.plot(x1, grafica1[i], label=labels[i])
        ax2a.set_title(titulo1)
        ax2a.legend()  
    if grafica2 is not None:
        x2 = np.arange(len(grafica2[0]))
        ax2b = plt.subplot2grid((2, 2), (1, 1))
        for i in range(1,3):
            ax2b.plot(x2, grafica2[i], label=labels[i])
        ax2b.set_title(titulo2)
        # ax2b.set_ylim(0, 3e4)
        ax2b.legend()
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    plt.show()
    
def guardar_imagen(archivo, imagen, deteccion, rutas, contador=None, rep=False):
    ruta = rutas[deteccion]
    ruta_completa = f"{ruta}/{archivo.split('.')[0]}_{contador}.jpg"
    cv2.imwrite(f"{ruta_completa}", imagen)

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

def distancia_valor(v, v_ideal=120):
    return round(100*(1-abs(v_ideal-v)/v_ideal))

def escalar(hora_float, valor_agua, valor_fondo):
    means = np.array([11.04201715, 104.49954614, 213.83933434])
    stds = np.array([3.70840555, 38.46259652, 15.74878125])
    medidas = [hora_float, valor_agua, valor_fondo]
    medidas_scaled = (medidas-means)/stds
    return np.array([medidas_scaled])

if __name__ == "__main__":
    
    print('\n')
    
    # Cargar las redes
    clasificador_agua = load_model("./weights/pesos_agua_facsa_17marzo.h5")
    clasificador_color = load_model("./weights/pesos_facsa_16marzo.h5")
    
    modo = int(input('¿Entrenamiento o test (0 - 1)?: '))
    modos = ['datos', 'test']
    
    # Abrir archivo CSV en modo de escritura
    with open(f'{modos[modo]}.csv', mode='w', newline='') as archivo_csv:
    # with open('validacion.csv', mode='w', newline='') as archivo_csv:

        # Crear objeto de escritura CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        
        # Encabezados
        datos = {"nombre": None, "valor_agua1": None, "valor_chorro1": None, "valor_fondo1": None, "valor_agua2": None, "valor_chorro2": None, "valor_fondo2": None, "hora": None, "minuto": None, "mes": None, "dia": None, "deteccion_red1": None, "deteccion_red2": None}
        
        # Escribir fila de encabezado
        escritor_csv.writerow(datos.keys())
        
        # Para medir el tiempo total
        t1 = time.time()
        
        # Borro los archivos de las carpetas de pruebas
        borrar = True
        if borrar:
            borrar_contenido(ruta_etiquetado)
            # borrar_contenido(ruta_buenestado)
            # borrar_contenido(ruta_muyclaro)
            # borrar_contenido(ruta_oscuro)
            # borrar_contenido(ruta_muyoscuro)
        
        # Contador de imagenes
        cont = 0
        
        # Recorro los archivos de la carpeta origen
        for archivo in tqdm(archivos):
            
            # DATOS QUE ME INTERESAN
            nombre = archivo.split('.')
            nombre = nombre[0].split('_')
            hora = nombre[-1]
            fecha = nombre[-2]
            hora_float = int(hora.split('-')[0]) + int(hora.split('-')[1])/60
            
            # Ruta de cada imagen
            src_file = os.path.join(ruta_origen, archivo)
                
            # Leer imagen
            imagen = cv2.imread(src_file)
            copia = imagen.copy()
            copia_fondo = imagen.copy()
            
            # ROIs
            tuberia1 = [360, 425, 775, 925]
            tuberia2 = [520, 585, 1225, 1375]
            
            # ROIs nuevos
            # tuberia1 = [360, 425, 815, 880]
            # tuberia2 = [520, 585, 1240, 1305]
            
            chorro1 = [550, 600, 775, 875]
            chorro2 = [600, 725, 1100, 1250]
            fondo1 = [220, 270, 900, 1000]
            fondo2 = [385, 435, 1380, 1445]
            imagen_tuberia1 = imagen[tuberia1[0]:tuberia1[1], tuberia1[2]:tuberia1[3]]
            imagen_tuberia2 = imagen[tuberia2[0]:tuberia2[1], tuberia2[2]:tuberia2[3]]
            imagen_chorro1 = imagen[chorro1[0]:chorro1[1], chorro1[2]:chorro1[3]]
            imagen_chorro2 = imagen[chorro2[0]:chorro2[1], chorro2[2]:chorro2[3]]
            imagen_fondo1 = imagen[fondo1[0]:fondo1[1], fondo1[2]:fondo1[3]]
            imagen_fondo2 = imagen[fondo2[0]:fondo2[1], fondo2[2]:fondo2[3]]
            # imagen_fondo = copia_fondo
            # imagen_fondo[fondo[0]:fondo[1], fondo[2]:fondo[3]] = 0

            # Cálculo de histogramas
            h, s, v = histograma(imagen, color='hsv', rep=False)
            h1, s1, v1 = histograma(imagen_tuberia1, color='hsv', rep=False)
            h2, s2, v2 = histograma(imagen_tuberia2, color='hsv', rep=False)
            hch1, sch1, vch1 = histograma(imagen_chorro1, color='hsv', rep=False)
            hch2, sch2, vch2 = histograma(imagen_chorro2, color='hsv', rep=False)
            hf1, sf1, vf1 = histograma(imagen_fondo1, color='hsv', rep=False)
            hf2, sf2, vf2 = histograma(imagen_fondo2, color='hsv', rep=False)
            
            # Cálculo de v
            v_modal_tub1 = calculo_v(v1, criterio=1)
            v_modal_tub2 = calculo_v(v2, criterio=1)
            v_modal_ch1 = calculo_v(vch1, criterio=1)
            v_modal_ch2 = calculo_v(vch2, criterio=1)
            v_modal_fondo1 = calculo_v(vf1, criterio=1)
            v_modal_fondo2 = calculo_v(vf2, criterio=1)
            tupla_v = (v_modal_tub1, v_modal_tub2)
            
            # # Peor valor de las dos tuberías
            # v_modal_tub, indice = peor_tuberia(tupla_v)
            
            # # Estados posibles
            estados = ["No nominal", "No nominal", "Nominal", "Exceso de poli", "Sin agua"]
            # rutas = [ruta_muyoscuro, ruta_oscuro, ruta_buenestado, ruta_muyclaro, ruta_sinagua]
            
            # # Comprobación de agua
            # agua1 = hay_agua(v_modal_tub1, v_modal_ch1, distancia_max=50)
            # agua2 = hay_agua(v_modal_tub2, v_modal_ch2, distancia_max=50)
            # Con red
            agua1 = round(clasificador_agua.predict(np.array([imagen_tuberia1]))[0][0])
            agua2 = round(clasificador_agua.predict(np.array([imagen_tuberia2]))[0][0])
            
            # # Clasificación de la imagen según v
            # deteccion1 = clasificacion(v_modal_tub1, v_modal_ch1)
            # deteccion2 = clasificacion(v_modal_tub2, v_modal_ch2)
            # detecciones = (deteccion1, deteccion2)
            # Con red
            if agua1:
                X1 = escalar(hora_float, v_modal_tub1, v_modal_fondo1)
                deteccion1 = round(clasificador_color.predict(X1)[0][0])
            else:
                deteccion1 = 4
            if agua2:
                X2 = escalar(hora_float, v_modal_tub2, v_modal_fondo2)
                deteccion2 = round(clasificador_color.predict(X2)[0][0])
            else:
                deteccion2 = 4
            
            # Colores
            colores_posibles = {"Rojo": (0, 0, 255), "Naranja": (0, 128, 255),"Amarillo": (255, 255, 0), "Verde": (0, 255, 0), "Azul": (255, 255, 0), "Blanco": (255, 255, 255)}
            colores = ["Rojo", "Naranja", "Verde", "Azul", "Blanco"]
            
            # Dibujo de ROIs
            cv2.rectangle(copia, (tuberia1[2], tuberia1[0]), (tuberia1[3], tuberia1[1]), colores_posibles[colores[deteccion1]], 2)
            cv2.rectangle(copia, (tuberia2[2], tuberia2[0]), (tuberia2[3], tuberia2[1]), colores_posibles[colores[deteccion2]], 2)
            # cv2.rectangle(copia, (chorro1[2], chorro1[0]), (chorro1[3], chorro1[1]), (0, 255, 0), 2)
            # # cv2.rectangle(copia, (chorro2[2], chorro2[0]), (chorro2[3], chorro2[1]), (0, 255, 0), 2)
            cv2.rectangle(copia, (fondo1[2], fondo1[0]), (fondo1[3], fondo1[1]), (255, 0, 255), 2)
            cv2.rectangle(copia, (fondo2[2], fondo2[0]), (fondo2[3], fondo2[1]), (255, 0, 255), 2)

            cv2.putText(copia, f"{v_modal_tub1}: {estados[deteccion1].upper()}", (tuberia1[2]-50, tuberia1[0]-120), cv2.FONT_HERSHEY_SIMPLEX, .9, colores_posibles[colores[deteccion1]], 2, cv2.LINE_AA)
            cv2.putText(copia, f"{v_modal_tub2}: {estados[deteccion2].upper()}", (tuberia2[2]-50, tuberia2[0]-200), cv2.FONT_HERSHEY_SIMPLEX, .8, colores_posibles[colores[deteccion2]], 2, cv2.LINE_AA)
            # # cv2.putText(copia, "Chorro_1", (chorro1[2], chorro1[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            # # cv2.putText(copia, "Chorro_2", (chorro2[2], chorro2[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            # # cv2.putText(copia, "Fondo", (fondo[2], fondo[0]-10), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 0, 255), 2)

            # # Guardado de las imágenes clasificadas
            # guardar_imagen(archivo, copia, min(deteccion1, deteccion2), rutas, cont)
            cv2.imwrite(f"{ruta_etiquetado}/{archivo}", copia)
            # cv2.imwrite(f"{ruta_presentacion}/{cont}", copia)        
            
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
                grafica2 = [hf1, sf1, vf1]
                presentacion(copia, grafica1, grafica2, titulo_imagen=f"Agua en tuberias -> 1: {estados[deteccion1]} 2: {estados[deteccion2]}", titulo1=f"Histograma tuberia 1, v = {v_modal_tub1}", titulo2=f"Histograma fondo, v = {v_modal_fondo}")
            
            # Imagen a pelo con detecciones
            presentar_cutre = False
            if presentar_cutre:
                presentacion_cutre(copia)#, titulo=f"Agua en tuberias    -->     1: {estados[deteccion1]}    2: {estados[deteccion2]}")
            
            # Contador de imagenes
            cont += 1
            
            # # Mostrar el proceso en consola
            # print(f"{cont}/{len(archivos)} - {archivo}")
            
            # # Alertas
            # if any(x != 2 for x in detecciones):
            #     mensaje = f"Alarma:\n\t- Tubería 1: {estados[deteccion1].upper()}\n\t- Tubería 2: {estados[deteccion2].upper()}\n"
            #     print(mensaje)
            #     # message = client.messages.create(body=mensaje,
            #     #                                 from_="+12706790926",
            #     #                                 to="+34652034697")
            # else:
            #     print("Todo correcto")
                
            # DATOS QUE ME INTERESAN
            datos1 = {"nombre": archivo, "valor_agua1": v_modal_tub1, "valor_chorro1": v_modal_ch1, "valor_fondo1": v_modal_fondo1, "valor_agua2": v_modal_tub2, "valor_chorro2": v_modal_ch2, "valor_fondo2": v_modal_fondo2, "hora": int(hora.split('-')[0]), "minuto": int(hora.split('-')[1]), "mes": int(fecha.split('-')[-2]), "dia": int(fecha.split('-')[-1]), "deteccion_red1": deteccion1, "deteccion_red2": deteccion2}
            # datos2 = {"nombre": archivo, "tuberia": 2, "valor_agua": v_modal_tub2, "valor_chorro": v_modal_ch2, "valor_fondo": v_modal_fondo2, "hora": int(hora.split('-')[0]), "minuto": int(hora.split('-')[1]), "mes": int(fecha.split('-')[-2]), "dia": int(fecha.split('-')[-1]), "deteccion_red": deteccion2}
            # print(list(datos.values()))
            
            # Escribir fila de datos
            escritor_csv.writerow(datos1.values())
            # escritor_csv.writerow(datos2.values())
            

    # Para medir el tiempo total
    t2 = time.time()
    print(f"Tiempo total = {round(t2-t1, 2)} segundos.")
