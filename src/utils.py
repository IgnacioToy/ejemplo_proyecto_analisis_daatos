import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import ExcepcionPersonalizada


def guardar_objeto(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise ExcepcionPersonalizada(e, sys)

def evaluar_modelo(X_train, y_train, X_test, y_test, models, param):
    try:
        reporte = {}

        for nombre_modelo, model in models.items():
            para = param.get(nombre_modelo, {})

            gs = GridSearchCV(model, para, cv= 3, n_jobs= 1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            reporte[nombre_modelo] = test_model_score

        return reporte
    
    except Exception as e:
        raise ExcepcionPersonalizada(e, sys)


def cargar_objeto(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise ExcepcionPersonalizada(e, sys)