import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRFRegressor

from src.exception import ExcepcionPersonalizada
from src.logger import logging
from src.utils import guardar_objeto, evaluar_modelo


@dataclass
class ConfigEntrenadorModelo:
    ruta_modelo_entrenado = os.path.join('artifacts', 'model.pkl')

class EntrenadorModelo:
    def __init__(self):
        self.config_entrenador_modelo = ConfigEntrenadorModelo()

    def iniciar_entrenador_modelo(self, entrenamiento_array, testeo_array):
        try:
            X_train, y_train, X_test, y_test = (
                entrenamiento_array[:, :-1], entrenamiento_array[:, -1],
                testeo_array[:, :-1], testeo_array[:, -1]
            )

        modelos = {
            'Random Forest': RandomForestRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Linear Regression': LinearRegression(),
            'XGBRegressor': XGBRFRegressor(),
            'CatBoosting Regressor': CatBoostRegressor(verbose=False),
            'AdaBost Regressor': AdaBoostRegressor(),
        }

        parametros = {
            "Decision Tree": {"criterion": ["squared_error", "friedman_mse"]},
            "Random Forest": {"n_estimators": [32, 64, 128]},
            "Gradient Boosting": {"learning_rate": [0.1, 0.01], "subsample": [0.6, 0.8]},
            "Linear Regression": {},
            "XGBRegressor": {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200]},
            "CatBoosting Regressor": {"depth": [6, 10], "learning_rate": [0.01, 0.05]},
            "AdaBoost Regressor": {"learning_rate": [0.1, 0.01], "n_estimators": [50, 100]}
        }

        reporte_modelos = evaluar_modelo(X_train, y_train, X_test, y_test, modelos, parametros)

        mejor_puntaje = max(sorted(reporte_modelos.values()))
        mejor_nombre_modelo = list(reporte_modelos.keys()[list(reporte_modelos.values()).index[mejor_puntaje]])
        mejor_modelo = modelos[mejor_nombre_modelo]

        if mejor_puntaje < 0.6:
            raise ExcepcionPersonalizada('Ningún modelo superó el umbral del 60%')

            logging.info(f'Mejor modelo: {mejor_nombre_modelo} con R2 {mejor_puntaje}')
            guardar_objeto(self.config_entrenador_modelo.ruta_modelo_entrenado, mejor_modelo)

            return mejor_puntaje
        
        except Exception as e:
            raise ExcepcionPersonalizada(e, sys)