# Segmentación y Conteo de Objetos

Proyecto de visión artificial para segmentar, detectar y contar objetos sobre una superficie de trabajo usando Python y OpenCV.

## Objetivo

Desarrollar una librería capaz de:

- Segmentar objetos
- Detectar bordes
- Detectar esquinas
- Contar objetos
- Obtener propiedades geométricas

## Técnicas implementadas

- Conversión RGB a gris
- Filtrado espacial
- Transformación de intensidad
- Thresholding
- Segmentación por regiones
- Watershed
- Detección de bordes (Canny)
- Detección de esquinas (Harris)

## Librerías usadas

- OpenCV
- NumPy
- Matplotlib

## Archivos del proyecto

- `vision_lib.py` → Librería principal
- `ejemplo.py` → Script de ejecución
- `imagenes_prueba/` → Imágenes utilizadas
- `resultados/` → Resultados obtenidos
- `reporte/` → Reporte técnico

## Resultados

El sistema permite detectar y contar objetos presentes en una imagen cenital mediante técnicas de procesamiento digital de imágenes.

## Limitaciones

- Sensible al contraste entre objeto y fondo.
- Puede fallar con objetos muy pegados.
- Sensible a sombras o iluminación irregular.

## Autores
LEONARDO DE LIRA ARTEAGA
BRIAN LEYVA DE LA TORRE
RAFAEL MEZA LOPEZ
RAFAEL WALDO JONGUITUD
