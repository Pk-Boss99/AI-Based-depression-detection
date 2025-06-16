# File: gui/components.py

import streamlit as st

# This file is for reusable Streamlit components.
# Add functions here to create custom UI elements that can be used across your app.

# Example of a simple reusable component function (currently not used in app.py)
def display_analysis_metric(label: str, value: float, help_text: str = None):
    """
    Displays a metric with a label and optional help text.
    """
    st.metric(label=label, value=f"{value:.2f}", help=help_text)

# You can add more functions here for things like:
# - Custom input fields with specific validation
# - Reusable display blocks for results
# - Navigation components (if building a multi-page app)
