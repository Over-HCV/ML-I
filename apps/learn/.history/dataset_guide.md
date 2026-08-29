# Microsoft GUIDE Dataset Guide

## Dataset Principal: Microsoft GUIDE

**Enlace de descarga**: `https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction`

## ¿Qué hace único a este dataset?

Microsoft está desafiando a la comunidad de ciencia de datos a desarrollar técnicas para **predecir el próximo incidente significativo de ciberseguridad**. GUIDE es la colección públicamente disponible más grande de incidentes reales de ciberseguridad.

## Características del Dataset

**¿Por Qué Este Dataset es Revolucionario?**
- **13+ millones** de evidencias reales de incidentes de ciberseguridad
- **33 tipos de entidades** (usuarios, dispositivos, IPs, dominios, etc.)
- **1.6 millones** de alertas y **1 millón** de incidentes anotados
- **6,100+ organizaciones** reales con telemetría auténtica  
- **441 técnicas MITRE ATT&CK** mapeadas
- **Período de 2 semanas** con resoluciones temporales detalladas

## Descarga del Dataset Principal

**Paso 1**: Ve a `https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction`

**Paso 2**: Crea cuenta gratuita en Kaggle (30 segundos)

**Paso 3**: Descarga el dataset (~2GB comprimido, ~15GB descomprimido)

## Estructura Esperada del Dataset

```
microsoft_guide/
├── incidents.csv              # Incidentes principales con triage labels
├── evidence.csv               # 13M+ evidencias por incidente  
├── alerts.csv                 # 1.6M alertas con DetectorIds
├── entities.csv               # 33 tipos de entidades
├── mitre_mapping.csv          # 441 técnicas MITRE ATT&CK
└── remediation_actions.csv    # 26K acciones de remediación
```

## Lo que Predecimos

1. **Probabilidad de incidente** en ventanas temporales (1h, 4h, 24h)
2. **Tipo de incidente** más probable (basado en MITRE ATT&CK)
3. **Impacto esperado** del incidente (basado en patrones históricos)
4. **Entidades afectadas** más probables
5. **Acciones de remediación** recomendadas

## Técnicas de Preprocessing para Datos de Incidentes

### Manejo de Datos Temporales Multi-Resolución

Los datos GUIDE tienen múltiples escalas temporales:
```python
# Diferentes resoluciones temporales en el dataset:
# - Evidencias: resolución por segundo/minuto
# - Alertas: resolución por minuto/hora  
# - Incidentes: resolución por hora/día
# - Patrones organizacionales: resolución semanal/mensual

# Necesitamos agregación inteligente para cada modelo:
# LSTM: secuencias horarias de features agregados
# GNN: snapshots de grafos cada 15 minutos  
# XGBoost: features agregados por ventana de predicción
```

### Balanceo de Clases para Incidentes Críticos

```python
# En SOCs reales: distribución extremadamente desbalanceada
# Incidentes normales: 95% de los datos
# Incidentes críticos: 2% de los datos  
# Incidentes APT avanzados: 0.1% ← ¡CRÍTICOS pero rarísimos!

# Estrategia de balanceo especializada:
# 1. SMOTE temporal para series de tiempo
# 2. Síntesis de grafos para datos relacionales
# 3. Sobremuestreo contextual por organización
```

## Instalación y Setup

1. Descargar dataset de Kaggle
2. Extraer archivos en `data/microsoft_guide/`
3. Ejecutar `uv sync` para instalar dependencias
4. Ejecutar `python src/data_adapters/microsoft_guide/validate_dataset.py` para verificar integridad