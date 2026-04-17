from ejjjjj import test_data_path
from src.components.data_transformation import TransformacionDatos
from src.components.entrenamiento_modelo import EntrenadorModelo

import pandas as pd

if __name__ == '__main__':
    # Rutas de datos
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    # Transformacion
    data_trans = TransformacionDatos()
    train_arr, test_arr, _ = data_trans.iniciar_transformacion_datos(train_data_path, test_data_path)
    
    # Entrenamiento
    model_trainer = EntrenadorModelo()
    resultado = model_trainer.iniciar_entrenador_modelo(train_arr, test_arr)
    print(f'Entrenamiento completado. Mejor R2: {resultado}')