from streamlit_option_menu import option_menu
import streamlit as st

def show_grouped_menu():
    # Create visual groupings using line separators and section labels
    with st.sidebar:
        st.markdown("### Overview")
        overview = option_menu(
            menu_title=None,
            options=["Dashboard", "Data Overview"],
            icons=["speedometer", "table"],
            default_index=0
        )

        st.markdown("---")
        st.markdown("### Analysis")
        analysis = option_menu(
            menu_title=None,
            options=["Insights", "Prediction Model"],
            icons=["bar-chart-line", "cpu"],
            default_index=0
        )

        st.markdown("### Doctor")
        doctor_tab = option_menu(
            menu_title=None,
            options=["Patient Info", "Patient Image", "Patient Notes", "Doctor Notes"],
            icons=["person", "image", "journal-text", "pencil-square"],
            default_index=0
        )

        st.markdown("---")
        st.markdown("### Business")
        business = option_menu(
            menu_title=None,
            options=["KPI Tracker", "Business Impact", "Recommendations"],
            icons=["graph-up-arrow", "cash-coin", "lightbulb"],
            default_index=0
        )

    # Combine all possible outputs and return the selected one
    return overview or analysis or business or doctor_tab

def show_menu():
    print("Menu goes here")
