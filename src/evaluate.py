import numpy as np
from sklearn.metrics import mean_squared_error

def compute_metrics(y_true, y_pred):
    """Retorna métricas determinísticas para modelos contínuos:
    - RMSE: penaliza resíduos de grande magnitude na escala do motor_UPDRS"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse))
    }
