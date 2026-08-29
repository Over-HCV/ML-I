## Plataforma Predictiva de Incidentes de Ciberseguridad

> **Documentación Modular**: Este proyecto está organizado en documentos especializados para mantener el principio de responsabilidad única.

### 📋 **Documentación del Proyecto**

- **[Visión del Proyecto](./project_overview.md)**: Visión general, objetivos y ventajas competitivas
- **[Dataset Microsoft GUIDE](./dataset_guide.md)**: Información detallada del dataset, descarga e instalación
- **[Arquitectura del Ensemble](./architecture_design.md)**: Diseño técnico de modelos y pipeline
- **[Métricas de Evaluación](./evaluation_metrics.md)**: Métricas especializadas para predicción de incidentes

### 🚀 **Quick Start**

1. **Clonar proyecto**:
   ```bash
   git clone <repository-url>
   cd cybersecurity-incident-predictor
   ```

2. **Instalar dependencias**:
   ```bash
   uv sync
   ```

3. **Descargar dataset** (ver [dataset_guide.md](./dataset_guide.md)):
   - Ve a Kaggle Microsoft GUIDE dataset
   - Descarga y extrae en `data/microsoft_guide/`

4. **Ejecutar dashboard**:
   ```bash
   uv run cybersec-dashboard
   ```

### 🔧 **Configuración del Proyecto**

- **Python**: 3.10+
- **Gestor de dependencias**: UV
- **Framework UI**: Solara  
- **ML Stack**: PyTorch, XGBoost, scikit-learn
- **Calidad de código**: Ruff, Black, MyPy

