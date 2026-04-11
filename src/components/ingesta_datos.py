import os
import sys
from src.exception import ExcepcionPersonalizada
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass