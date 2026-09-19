"""
Page 1: Home & Setup
Project overview and quick start guide
"""

import streamlit as st
import streamlit.components.v1 as components


def render():
    """Main render function for Home page"""
    st.title("Machine Learning & Deep Learning")

    render_project_overview()
    st.divider()
    render_workflow_guide()


def render_project_overview():
    """Section 1: What this app does with workflow diagram"""
    st.header("Project Overview")

    st.markdown("""
    Build and evaluate deep learning models for malware image classification:
    - Configure dataset splits and augmentation
    - Design CNN, Transformer, or Transfer Learning architectures
    - Train with customizable hyperparameters
    - Analyze results with advanced metrics and visualizations
    """)

    # Workflow diagram with improved Mermaid
    mermaid_code = """
    flowchart LR
        subgraph Config["⚙️ Configuration"]
            D["📁 Dataset"]
            M["🧠 Model"]
            T["🎯 Training"]
        end
        subgraph Run["▶️ Execution"]
            Mo["📊 Monitor"]
        end
        subgraph Analysis["📈 Analysis"]
            R["✅ Results"]
            I["🔍 Interpret"]
        end
        D --> M --> T --> Mo --> R --> I
    """

    components.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
        <div class="mermaid" style="display: flex; justify-content: center;">{mermaid_code}</div>
        """,
        height=180,
    )


def render_workflow_guide():
    """Section 2: Step-by-step workflow guide"""
    st.header("Quick Start")

    st.markdown("""
    **New Training Session:**
    1. Click **"New Session"** in the header to start fresh
    2. **Dataset** → Configure data splits and augmentation
    3. **Model** → Design your neural network architecture
    4. **Training** → Set hyperparameters (optimizer, epochs, etc.)
    5. **Monitor** → Start training and watch progress
    6. **Results** → Evaluate with confusion matrix, per-class metrics
    7. **Interpret** → Visualize model attention with Grad-CAM

    **Resume Previous Work:**
    - Use **"Past Sessions"** dropdown in the header to load saved sessions
    """)
