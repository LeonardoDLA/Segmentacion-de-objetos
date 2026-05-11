
import cv2
import numpy as np


class ContadorObjetos:

    def __init__(self):
        pass

    
    # PREPROCESAMIENTO
    def mejorar_imagen(self,imagen):

        imagen_gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

        # Suavizar ruido
        imagen_suave = cv2.GaussianBlur(imagen_gris,(5,5),0)

        # Mejorar contraste
        imagen_mejorada = (cv2.convertScaleAbs(imagen_suave,alpha=1.15,beta=5))

        return imagen_mejorada

    # SEGMENTACIÓN
    def separar_objetos(self,imagen_gris):

        # Threshold automático
        _, mascara = cv2.threshold(imagen_gris,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)

        # Limpiar ruido
        mascara = cv2.morphologyEx(mascara,cv2.MORPH_OPEN,kernel,iterations=1)

        # Cerrar huecos
        mascara = cv2.morphologyEx(mascara,cv2.MORPH_CLOSE,kernel,iterations=2)

        return mascara

    # BORDES
    def sacar_bordes(self,imagen_gris):

        bordes = cv2.Canny(imagen_gris,50,150)

        return bordes

    # ESQUINAS
    def sacar_esquinas(self,imagen_gris,mascara):

        solo_objetos = (cv2.bitwise_and(imagen_gris,imagen_gris,mask=mascara))

        esquinas = cv2.cornerHarris(np.float32(solo_objetos),2,3,0.04)

        return esquinas

    # CONTEO
    def contar_objetos(self,mascara,imagen_original):

        imagen_resultado = (imagen_original.copy())

        datos_objetos = []

        # Watershed para separar objetos pegados
        kernel = np.ones((3,3),np.uint8)

        fondo_seguro = cv2.dilate(mascara,kernel,iterations=2)

        distancia = (cv2.distanceTransform(mascara,cv2.DIST_L2,5))

        _, primer_plano = (cv2.threshold(distancia,0.25 * distancia.max(),255,0))

        primer_plano = np.uint8(primer_plano)

        zona_desconocida = cv2.subtract(fondo_seguro,primer_plano)

        _, marcadores = (cv2.connectedComponents(primer_plano))

        marcadores = (marcadores + 1)

        marcadores[zona_desconocida == 255] = 0

        marcadores = cv2.watershed(imagen_original,marcadores)

        numero_objeto = 1

        for etiqueta in np.unique(marcadores):

            if etiqueta <= 1:
                continue

            mascara_objeto = np.zeros(mascara.shape,dtype="uint8")

            mascara_objeto[marcadores == etiqueta] = 255

            contornos, _ = (cv2.findContours(mascara_objeto,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE))

            if len(contornos) == 0:
                continue

            contorno = max(contornos,key=cv2.contourArea)

            area = cv2.contourArea(contorno)

            # eliminar ruido
            if area < 300:
                continue

            x, y, ancho, alto = (cv2.boundingRect(contorno))

            perimetro = (cv2.arcLength(contorno,True))

            momentos = (cv2.moments(contorno))

            if momentos["m00"] != 0:

                centro_x = int(momentos["m10"]/momentos["m00"])

                centro_y = int(momentos["m01"]/momentos["m00"])

            else:

                centro_x = 0
                centro_y = 0

            cv2.drawContours(imagen_resultado,[contorno],-1,(0,255,0),2)

            cv2.circle(imagen_resultado,(centro_x,centro_y),5,(255,0,0),-1)

            cv2.putText(imagen_resultado,f"#{numero_objeto}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            datos_objetos.append({

                "numero":numero_objeto,
                "area":area,
                "perimetro":perimetro,
                "centro":(centro_x,centro_y)

            })

            numero_objeto += 1

        return (imagen_resultado,datos_objetos)
