import streamlit as st

# Class to generate multiple Streamlit pages using an object oriented approach 
class MultiPage: 
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="💰"
        )
    
    def add_page(self, title, func) -> None: 
        """Add a new page to the app."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title(self.app_name) # put app name in sidebar not in the middle
        page = st.sidebar.radio('Navigation', self.pages, format_func=lambda page: page['title'])
        page['function']()