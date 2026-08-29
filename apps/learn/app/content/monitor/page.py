"""
Page: Training Monitor
Self-contained page module
"""

from apps.learn.app.components.header import render_header
from apps.learn.app.components.sidebar import render_sidebar
from apps.learn.app.content.monitor.view import render

render_header()
render_sidebar()

render()
