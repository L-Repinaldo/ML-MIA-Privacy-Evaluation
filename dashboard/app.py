from navigation import view_router

import streamlit as st


def main():
    st.set_page_config(
        page_title="ML Privacy Trade-off Explorer",
        layout="wide",
    )

    view_router.render_selected_view()


if __name__ == "__main__":
    main()