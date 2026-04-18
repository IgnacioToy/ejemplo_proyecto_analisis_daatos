import os
import sys
from src.exception import ExcepcionPersonalizada
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.transformacion_datos import TransformacionDatos
from src.components.model_trainer import EntrenadorModelo

@dataclass
class ConfigIngestaDatos:
    # Definamos donde se guardaran los archivos temporales de datos
    ruta_entrenamiento: str = os.path.join('artifacts', 'train.csv')
    ruta_prueba: str = os.path.join('artifacts', 'test.csv')
    ruta_datos_brutos: str = os.path.join('artifacts', 'data.csv')

class IngestaDatos:
    def __init__(self):
        self.config_ingesta = ConfigIngestaDatos()

    def iniciar_ingesta_datos(self):
        logging.info('Iniciando el componente de ingesta de datos')
        try:
            # Leer el dataset original (ajusta la ruta según el proyecto)
            df = pd.read_csv('nebook/ data / stud.csv')
            logging.info('Dataset leído correctamente como DataFrame.')

            # Crear carpeta artifacts si no existe
            os.makedirs(os.path.dirname(self.config_ingesta.ruta_entrenamiento), exist_ok=True)

            # Guardado el bruto
            df.to_csv(self.config_ingesta.ruta_datos_brutos, index= False, header=True)

            logging.info('Iniciando la división Train Test Split')
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state= 42)

            # Guardo train y test por separado
            train_set.to_csv(self.config_ingesta.ruta_entrenamiento, index= False, header= True)
            test_set.to_csv(self.config_ingesta.ruta_datos_brutos, index= True, header= True)

            logging.info('Ingesta de datos finalizada exitosamente')

            return (
                self.config_ingesta.ruta_entrenamiento,
                self.config_ingesta.ruta_prueba
            )

        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)

# Bloque de ejecución principal para probar todo el flujo
if __name__ == '__main__':
    # Ingesta
    obj = IngestaDatos()
    entrenamiento_datos, testeo_datos = obj.iniciar_ingesta_datos()

    # Transformacion
    transformacion_datos = TransformacionDatos()
    entrenamiento_arr, test_arr, _ = data_transformation.iniciar_transformacion_datos(train_data, test_data)

    # Entrenamiento 
    entrenamiento_modelo = EntrenadorModelo()
    print(f'R2 del mejor modelo:{entrenamiento_modelo.iniciar_entrenador_modelo(entrenamiento_arr, testeo_datos)}')