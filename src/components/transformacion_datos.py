import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import ExcepcionPersonalizada
from src.logger import logging
from src.utils import guardar_objeto

@dataclass
class ConfigTransfDatos:
    archivo_procesado = os.path.join('artifacts', 'preprocessor.pkl')

class TransformacionDatos:
    def __init__(self):
        self.config_transformacion_datos = ConfigTransfDatos()

    def obtener_transformador_datos(self):
        try:
            columnas_numericas = ['writing_score', 'reading_score']
            columnas_categoricas = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course'
            ]

            # Pipeline Numérico
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Pipeline Categórico (Corregido a 'most_frequent')
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Creando ColumnTransformer para pipelines.')

            # El procesador DEBE estar dentro del try e indentado correctamente
            procesador = ColumnTransformer([
                ('num_pipeline', num_pipeline, columnas_numericas),
                ('cat_pipeline', cat_pipeline, columnas_categoricas)
            ])

            return procesador
        
        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)

    def iniciar_transformacion_datos(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Lectura de datos completada')
            objetivo_procesamiento = self.obtener_transformador_datos()

            columna_objetivo = 'math_score'

            input_feature_train_df = train_df.drop(columns=[columna_objetivo], axis=1)
            target_feature_train_df = train_df[columna_objetivo]

            input_feature_test_df = test_df.drop(columns=[columna_objetivo], axis=1)
            target_feature_test_df = test_df[columna_objetivo]

            logging.info('Aplicando preprocesamiento...')

            # fit_transform en train / transform en test
            input_feature_train_arr = objetivo_procesamiento.fit_transform(input_feature_train_df)
            input_feature_test_arr = objetivo_procesamiento.transform(input_feature_test_df)

            # Reconstrucción de arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Guardado corregido
            guardar_objeto(
                file_path=self.config_transformacion_datos.archivo_procesado,
                obj=objetivo_procesamiento
            )

            return (
                train_arr,
                test_arr,
                self.config_transformacion_datos.archivo_procesado,
            )
        
        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)