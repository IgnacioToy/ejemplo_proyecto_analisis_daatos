import sys
import os
import pandas as pd
from src.exception import ExcepcionPersonalizada
from src.logger import logging
from src.utils import cargar_objeto

class PredictPipeline:
    def __init__(self):
        pass

    def predecir(self, features):
        try:
            modelo_path = os.path.join("artifacts", "model.pkl")
            preprocesador_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Cargando modelo y preprocesador...")
            modelo = cargar_objeto(file_path=modelo_path)
            preprocesador = cargar_objeto(file_path=preprocesador_path)

            datos_escalados = preprocesador.transform(features)
            prediccion = modelo.predict(datos_escalados)
            return prediccion
        
        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)

class DatosPersonalizados:
    def __init__(self, 
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def obtener_dataframe(self):
        try:
            dict_personalizado = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(dict_personalizado)
        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)