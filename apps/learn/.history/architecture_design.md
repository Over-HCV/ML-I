# Arquitectura del Ensemble Híbrido

## Arquitectura del Ensemble Híbrido para Predicción de Incidentes

### Nivel 0: Modelos Base Especializados

#### 1. LSTM/GRU (Modelos Temporales)
```python
# ¿Por qué para predicción de incidentes?
# Los incidentes siguen patrones temporales:
# - "Después de 3 alertas de login fallido en 10min → 85% prob. de brute force en 30min"
# - "Secuencia: scan_ports → lateral_movement → data_exfil (en 2-6 horas)"
# - "Patrón organizacional: incident_spike cada lunes 9AM (parches de sistema)"
```
**Razón técnica**: LSTM aprende dependencias temporales de largo plazo en secuencias de evidencias, capturando cómo eventos pasados influyen en incidentes futuros.

#### 2. Graph Neural Network (Modelos de Relaciones)
```python
# ¿Por qué GNN para incidentes?
# Las 33 entidades forman grafos complejos:
# - user_A → compromised → accesses → server_B → contains → sensitive_data
# - IP_externa → scans → firewall → triggers → multiple_alerts → across → subnets
# - domain_controller → anomaly → propagates_to → workstations → indicates → lateral_movement
```
**Razón técnica**: GNN modela cómo el riesgo se propaga a través de la topología de red y relaciones entre entidades.

#### 3. XGBoost (Reconocimiento de Patrones en Alertas)
```python
# ¿Por qué XGBoost para evidencias?
# Analistas SOC piensan en reglas complejas:
# "Si detector_X + severity_high + entity_type_user + time_weekend → 92% incident"
# "Si MITRE_technique T1078 + multiple_orgs + similar_IOCs → coordinated_attack"
```
**Razón técnica**: XGBoost aprende reglas complejas de decisión sobre features de alertas y evidencias, optimizando para métricas específicas del negocio.

#### 4. Transformer (Secuencias de Evidencias)
```python
# ¿Por qué Transformer para cadenas de evidencias?
# Los incidentes son SECUENCIAS de evidencias con dependencias complejas:
# [evidence_1: port_scan] → [evidence_2: failed_auth] → [evidence_3: privilege_escalation]
# La atención del transformer aprende QUÉ evidencias predicen QUÉ incidentes
```
**Razón técnica**: Self-attention mechanisms capturan relaciones entre evidencias no-consecutivas que son críticas para predicción de incidentes.

### Nivel 1: Meta-Ensemble Adaptativo - El Director de SOC Virtual

Imagina un **Director de SOC** que coordina 4 analistas especializados:

```python
# Analista Temporal (LSTM): "Basándome en patrones históricos, 90% prob. de incident en 4h"
# Analista de Relaciones (GNN): "El riesgo se está propagando por la red, 85% prob."
# Analista de Reglas (XGBoost): "Las alertas actuales coinciden con pattern de credential_theft, 95% prob."
# Analista de Secuencias (Transformer): "La secuencia de evidencias indica escalada en 2h, 88% prob."

# El Meta-Ensemble aprende patrones como:
# "Cuando XGBoost y Transformer concuerdan fuertemente → confianza alta"
# "Cuando LSTM predice bajo riesgo pero GNN alto → revisar propagación lateral"
# "Para organizaciones financieras: confiar más en patrones temporales"
```

### Nivel 2: Ensemble Dinámico con Pesos Contextuales

```python
# Pesos no fijos, sino adaptativos por contexto:
def adaptive_weights(context):
    if context['organization_type'] == 'financial':
        # Sector financiero: temporales más importantes
        return [0.4, 0.2, 0.2, 0.2]  # LSTM, GNN, XGBoost, Transformer
    elif context['incident_history'] == 'APT_previous':
        # Historial de APT: relaciones más importantes  
        return [0.2, 0.4, 0.2, 0.2]  # GNN dominante
    elif context['time_of_day'] == 'weekend':
        # Fines de semana: secuencias más importantes
        return [0.2, 0.2, 0.2, 0.4]  # Transformer dominante
    
# El sistema aprende automáticamente estos patrones contextuales
```

## Pipeline de Predicción de Incidentes - Flujo Determinístico

```python
# PASO 1: Ingesta adaptativa de datos
from cybersec_platform.data_adapters import detect_dataset_type
dataset_adapter = detect_dataset_type(raw_data)  # Auto-detecta: GUIDE, CIC-IDS2017, etc.
processed_data = dataset_adapter.preprocess(raw_data)

# PASO 2: Feature engineering multi-modal  
features = {
    'temporal': extract_temporal_sequences(processed_data, window_size='4h'),
    'graph': build_entity_graph(processed_data, entity_types=33),
    'evidence': extract_evidence_features(processed_data, mitre_mapping=True),
    'contextual': extract_organizational_context(processed_data)
}

# PASO 3: Ensemble training pipeline
ensemble_trainer = AdaptiveEnsembleTrainer()
models = {
    'temporal_lstm': LSTMIncidentPredictor(sequence_length=24),  # 24h de historia
    'graph_gnn': GraphIncidentPredictor(node_types=33),
    'evidence_xgb': XGBoostIncidentPredictor(mitre_features=True), 
    'sequence_transformer': TransformerIncidentPredictor(attention_heads=8)
}

# PASO 4: Meta-ensemble con pesos contextuales
meta_ensemble = ContextualMetaEnsemble(
    base_models=models,
    context_features=['org_type', 'time_of_day', 'incident_history'],
    adaptation_strategy='online_learning'
)

# PASO 5: Entrenamiento con validación temporal
# Crítico: no usar validación cruzada normal, sino temporal split
train_data, val_data, test_data = temporal_train_test_split(
    features, split_dates=['2024-01-15', '2024-01-20']  # Respeta orden temporal
)

meta_ensemble.fit(train_data, validation_data=val_data)

# PASO 6: Predicción en tiempo real  
incident_predictions = meta_ensemble.predict_incidents(
    live_data, 
    prediction_horizons=['1h', '4h', '24h'],
    confidence_threshold=0.85
)

# PASO 7: Dashboard actualización automática
dashboard.update_real_time(incident_predictions)
```

## Arquitectura del Proyecto - Modular y Adaptable

```
cybersecurity_incident_predictor/
├── src/
│   ├── data_adapters/           # Adapters para diferentes datasets
│   ├── models/                 # Modelos especializados  
│   ├── evaluation/             # Métricas especializadas
│   ├── dashboard/              # Interfaz de usuario
│   ├── pipeline/               # Orquestación
│   └── utils/                  # Utilidades
├── config/                     # Configuraciones
├── tests/                      # Tests comprehensivos
├── deployment/                 # Docker, K8s, etc.
├── docs/                      # Documentación
├── pyproject.toml            # Configuración del proyecto con UV
└── README.md                 # Getting started
```