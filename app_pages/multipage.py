"""
Defines the MultiPage class to create a Streamlit application with multiple pages.

This module provides a simple object-oriented framework to register multiple 
pages in a Streamlit app and navigate between them using a sidebar menu.
"""

import streamlit as st

class MultiPage:
    """
    Class to manage multiple Streamlit pages in a single app.

    Attributes:
        pages (list): A list of dictionaries, each containing a page title and function.
        app_name (str): The name of the application, used in the sidebar and page title.
    """

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="💰"
        )
    
    def add_page(self, title, func) -> None:
        """
        Register a new page in the app.

        Args:
            title (str): The title of the page to display in the sidebar.
            func (callable): The function that renders the page content.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Display the sidebar navigation and render the selected page.
        """
        st.sidebar.title(self.app_name)
        page = st.sidebar.radio('Navigation', self.pages, format_func=lambda page: page['title'])
        page['function']()
