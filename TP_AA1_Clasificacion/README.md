# Predicción de Lluvias - Instrucciones

Este proyecto predice si lloverá o no al día siguiente utilizando un modelo de machine learning. Para ejecutar el proyecto, sigue las instrucciones detalladas a continuación.

## Requisitos previos

- **Docker** debe estar instalado y funcionando en tu máquina (si tenés Windows: https://docs.docker.com/desktop/setup/install/windows-install/).
- Tener acceso a una terminal o consola de comandos.

## 1. Clonar el repositorio

Primero, clona este repositorio en tu máquina local utilizando el siguiente comando:

git clone https://github.com/milarizzotto/AA1-TUIA-Casado-Rizzotto.git

## 2. Construir la imagen Docker
Accede al directorio donde hayas clonado este repositorio y accedé a la carpeta 'TP_AA1_Clasificacion'

Desde esa posición, construye la imagen Docker con el siguiente comando:

docker build -t tp-aa1-casado-rizzotto ./docker

Este comando creará una imagen Docker llamada tp-aa1-casado-rizzotto utilizando el Dockerfile ubicado en la carpeta ./docker.

## 3. Ejecutar el contenedor Docker
Una vez que la imagen Docker esté construida, ejecuta el contenedor Docker con el siguiente comando:

docker run -it --rm -v "TU_UBICACION_ABSOLUTA_DE_CARPETA_FILES:/files" tp-aa1-casado-rizzotto bash

### Importante:
Reemplaza "TU_UBICACION_ABSOLUTA_DE_CARPETA_FILES" con la ruta absoluta de la carpeta files en tu máquina local.

Por ejemplo, si tienes la carpeta files en la carpeta C:/Users/tuUsuario/OneDrive/Escritorio/TP_AA1_Clasificacion, el comando sería:

docker run -it --rm -v "C:/Users/tuUsuario/OneDrive/Escritorio/TP_AA1_Clasificacion/files:/files" tp-aa1-casado-rizzotto bash
Esto montará la carpeta local files dentro del contenedor y te permitirá acceder a ella dentro del contenedor Docker.

## 4. Ejecutar la inferencia
Una vez dentro del contenedor Docker, ejecuta el siguiente comando para realizar las predicciones:

python inference.py
Este comando procesará el archivo input.csv en la carpeta /files y generará un archivo de salida llamado output.csv con las predicciones del modelo.

## 5. Archivos de entrada y salida
input.csv
El archivo input.csv debe contener los datos de entrada para el modelo y debe ser colocado en la carpeta que montaste en el contenedor (files). Un archivo input.csv de ejemplo se provee en el repositorio.

output.csv
El archivo output.csv será generado en la misma carpeta files después de ejecutar el comando python inference.py. Este archivo contendrá las predicciones del modelo, con valores de 0 (no lloverá) o 1 (sí lloverá).

Ejemplo de cómo lucirá el archivo output.csv:

Rain_Prediction  
0  
1  
0  
1  
...  
Cada fila del archivo output.csv corresponderá a una predicción para cada entrada en el archivo input.csv.
