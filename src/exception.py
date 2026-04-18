import sys
from src.logger import logging

def mensaje_error_detalle(error, detalle_error):
    _, _, exc_tb = detalle_error.exc_info()
    nombre_archivo = exc_tb.tb_frame.f_code.co_filename
    mensaje_error = 'El error ocurrió en Python src: [{0}]. Línea numero: [{1}]. Mensaje error: [{2}]'.format(
        nombre_archivo, exc_tb.tb_lineno, str(error)
    )
    return mensaje_error

class ExcepcionPersonalizada(Exception):
    def __init__(self, mensaje_error, detalle_error):
        super().__init__(mensaje_error)
        self.mensaje_error = mensaje_error_detalle(mensaje_error, detalle_error=detalle_error)

    def __str__(self):
        return self.mensaje_error

