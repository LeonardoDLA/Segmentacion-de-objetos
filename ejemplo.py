import matplotlib.pyplot as plt
from vision_lib import ContadorObjetos
import cv2
import os

nombre_imagen = ("imagenes_prueba/negro2.jpg")

if not os.path.exists(nombre_imagen):
    print("No se encontró la imagen")

else:

    foto = cv2.imread(nombre_imagen)

    contador = (ContadorObjetos())

    # PREPROCESAMIENTO
    imagen_lista = (contador.mejorar_imagen(foto))

    # SEGMENTACIÓN
    mascara_final = (contador.separar_objetos(imagen_lista))

    # BORDES
    bordes = (contador.sacar_bordes(imagen_lista))

    # ESQUINAS
    esquinas = (contador.sacar_esquinas(imagen_lista,mascara_final))

    imagen_esquinas = (foto.copy())

    limite = (0.05 * esquinas.max())

    imagen_esquinas[esquinas > limite] = [255, 0, 0]

    # CONTEO
    imagen_final, datos = (contador.contar_objetos(mascara_final,foto))

    # MOSTRAR RESULTADOS
    plt.figure(figsize=(18, 10))
    plt.subplot(2,3,1)
    plt.title("Imagen Original")
    plt.imshow(cv2.cvtColor(foto,cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.title("Preprocesamiento")
    plt.imshow(imagen_lista,cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.title("Segmentación")
    plt.imshow(mascara_final,cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.title("Detección de Bordes")
    plt.imshow(bordes,cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.title("Detección de Esquinas")
    plt.imshow(cv2.cvtColor(imagen_esquinas,cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.title("Resultado Final")
    plt.imshow(cv2.cvtColor(imagen_final,cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # RESULTADOS
    print("-" * 40)
    print("RESULTADOS")
    print("-" * 40)
    print("Objetos encontrados:",len(datos))
    print("-" * 40)

    for objeto in datos:

        print(f"Objeto " f"#{objeto['numero']}")
        print(f"Área: " f"{objeto['area']:.2f}")
        print(f"Perímetro: " f"{objeto['perimetro']:.2f}")
        print(f"Centro: "f"{objeto['centro']}")
        print("-" * 20)
