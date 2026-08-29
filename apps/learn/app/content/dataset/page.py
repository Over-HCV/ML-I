"""
Page: Dataset Configuration
Self-contained page module
"""

from components.header import render_header
from components.sidebar import render_sidebar
from content.dataset.view import render

# Render persistent components
render_header()
render_sidebar()

# Render page content
render()
