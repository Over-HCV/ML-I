# Métricas de Evaluación para Predicción de Incidentes

## ¿Por Qué las Métricas Tradicionales No Bastan?

La **predicción de incidentes** requiere métricas completamente diferentes a la clasificación:

```python
# En un SOC predictivo:
# - PREDICCIÓN TEMPRANA EXITOSA = Incidente prevenido = $$$ ahorrados
# - FALSO NEGATIVO = Incidente no predicho = Impacto total del incidente  
# - FALSO POSITIVO = Recursos asignados innecesariamente = Costo de oportunidad

# Complejidad temporal:
# - Predecir en 1 hora = Muy valioso pero difícil
# - Predecir en 24 horas = Menos valioso pero más factible  
# - Predecir después del incidente = Inútil
```

## Métricas Temporales (Específicas para Predicción)

### 1. Mean Time to Detection (MTTD) Predictivo
```python
# ¿Cuánto tiempo ANTES del incidente lo predijimos?
mttd_predictive = incident_time - first_prediction_time
# Ejemplo: Incidente a las 14:00, primera predicción a las 10:00 → MTTD = 4 horas
```

### 2. Precision/Recall por Horizonte Temporal
```python
# Precisión varía dramáticamente por ventana de predicción:
precision_1h = 0.95    # Muy preciso a corto plazo
precision_4h = 0.87    # Menos preciso pero más tiempo para actuar
precision_24h = 0.65   # Menos preciso, pero permite planeación estratégica
```

### 3. Early Warning Score
```python
# Combina precisión con utilidad temporal
early_warning_score = precision * (prediction_horizon / max_useful_horizon)
# Premia predicciones tempranas pero precisas
```

## Métricas de Negocio (Críticas para SOCs)

### 1. Cost-Weighted Accuracy
```python
# No todos los incidentes cuestan igual:
incident_costs = {
    'data_breach': 4_000_000,      # $4M promedio
    'ransomware': 1_500_000,       # $1.5M promedio  
    'ddos': 100_000,               # $100K promedio
    'phishing': 10_000             # $10K promedio
}

cost_weighted_recall = sum(recall[incident] * cost[incident] for incident in types) / total_cost
```

### 2. Alert Fatigue Prevention
```python
# Precisión ponderada por volumen de alertas
alert_fatigue_score = precision / (predictions_per_day / analyst_capacity)
# Penaliza modelos que generan demasiadas alertas
```

### 3. Remediation Action Accuracy  
```python
# ¿Qué tan seguido las acciones recomendadas son correctas?
remediation_accuracy = correct_actions / total_recommended_actions
```

## Métricas de Adaptabilidad del Sistema

### 1. Cross-Organization Generalization
```python
# ¿Qué tan bien el modelo entrenado en org_A predice incidentes en org_B?
cross_org_performance = evaluate_model(trained_on='org_A', tested_on='org_B')
```

### 2. Model Drift Detection  
```python
# ¿Cuándo degrada el rendimiento del modelo?
drift_detection = performance_today / performance_baseline < 0.9
```

### 3. Few-Shot Learning Capability
```python
# ¿Puede adaptarse a nuevos tipos de incidentes con pocos ejemplos?
few_shot_accuracy = train_on_k_examples(k=10) → test_accuracy
```

## Implementación de Métricas Personalizadas

```python
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class CyberSecurityMetrics:
    """Custom metrics for cybersecurity incident prediction."""
    
    def __init__(self, incident_costs: Dict[str, float]):
        self.incident_costs = incident_costs
    
    def predictive_mttd(self, 
                       predictions: List[Tuple[float, str]], 
                       actual_incidents: List[Tuple[float, str]]) -> float:
        """Calculate Mean Time to Detection for predictive models."""
        total_early_detection_time = 0
        early_detections = 0
        
        for pred_time, pred_type in predictions:
            for incident_time, incident_type in actual_incidents:
                if pred_type == incident_type and pred_time < incident_time:
                    total_early_detection_time += (incident_time - pred_time)
                    early_detections += 1
                    break
        
        return total_early_detection_time / max(early_detections, 1)
    
    def cost_weighted_recall(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           labels: List[str]) -> float:
        """Calculate cost-weighted recall based on incident severity."""
        _, recall_per_class, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(labels)), average=None
        )
        
        total_cost = sum(self.incident_costs.get(label, 1.0) for label in labels)
        weighted_recall = sum(
            recall_per_class[i] * self.incident_costs.get(labels[i], 1.0) 
            for i in range(len(labels))
        ) / total_cost
        
        return weighted_recall
    
    def alert_fatigue_score(self, 
                          precision: float, 
                          predictions_per_day: int, 
                          analyst_capacity: int = 100) -> float:
        """Calculate alert fatigue prevention score."""
        return precision * min(1.0, analyst_capacity / max(predictions_per_day, 1))
```

## Evaluación Temporal Específica

```python
class TemporalEvaluator:
    """Evaluator that respects temporal order in cybersecurity data."""
    
    def temporal_train_test_split(self, 
                                data: pd.DataFrame, 
                                split_ratio: float = 0.8) -> Tuple:
        """Split data respecting temporal order."""
        data_sorted = data.sort_values('timestamp')
        split_index = int(len(data_sorted) * split_ratio)
        
        train_data = data_sorted.iloc[:split_index]
        test_data = data_sorted.iloc[split_index:]
        
        return train_data, test_data
    
    def sliding_window_evaluation(self, 
                                model, 
                                data: pd.DataFrame, 
                                window_size: str = '1D') -> Dict:
        """Evaluate model performance using sliding time windows."""
        results = []
        
        for window_start in pd.date_range(
            start=data['timestamp'].min(), 
            end=data['timestamp'].max(), 
            freq=window_size
        ):
            window_end = window_start + pd.Timedelta(window_size)
            window_data = data[
                (data['timestamp'] >= window_start) & 
                (data['timestamp'] < window_end)
            ]
            
            if len(window_data) > 0:
                predictions = model.predict(window_data)
                metrics = self.calculate_metrics(window_data, predictions)
                results.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'metrics': metrics
                })
        
        return results
```