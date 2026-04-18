from src.pipeline.predict_pipeline import PredictPipeline, DatosPersonalizados

# Creo los datos de un alumno inventado
alumno = DatosPersonalizados(
    gender='female',
    race_ethnicity='grupo B',
    parental_level_of_education= "bachelor's degree",
    lunch= 'standard',
    test_preparation_course= 'none',
    reading_score= 72,
    writing_score= 74
)

# Convertimos a DataFrame
df_alumno = alumno.obtener_dataframe()

# LLamo al pipeline de prediccion
pipeline = PredictPipeline()
resultado = pipeline.predecir(df_alumno)

print(f'La nota de matemátias estimada es: {resultado[0]:.2f}')