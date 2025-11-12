"""
Reusable plotting utilities.

This module defines reusable visualization functions used across multiple
pages of the app to ensure consistent, interactive, and interpretable
data exploration.
"""

import streamlit as st
import plotly.express as px


def plot_categorical(df, col, target_var):
    # Create interactive countplot
    fig = px.histogram(
        df,
        x=col,
        color=target_var,
        barmode="group",  # shows bars side by side like hue in seaborn
        category_orders={col: df[col].value_counts().index.tolist()},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        title_text=f"{col}",
        title_x=0.5,
        xaxis_title="",
        yaxis_title="Count",
        legend_title=target_var
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_numerical(df, col, target_var):
    fig = px.histogram(
        df,
        x=col,
        color=target_var,
        barmode="overlay",       # overlay bars for each target class
        histnorm='',             # raw counts, can also use 'percent'
        marginal="box",
        color_discrete_sequence=px.colors.qualitative.Set2,
        nbins=50                # adjust for resolution
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color="black"
    )

    fig.update_layout(
        title_text=f"{col}",
        title_x=0.5,
        xaxis_title="",
        yaxis_title="Count",
        legend_title=target_var
    )

    st.plotly_chart(fig, use_container_width=True)
