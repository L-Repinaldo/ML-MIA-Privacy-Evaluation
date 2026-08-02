import streamlit as st

def utility(utility_metrics):
     with st.expander("Utility Metrics"):
            st.dataframe(
                utility_metrics,
                use_container_width=True,
                hide_index=True,
            )


def attack(attack_metrics): 
      with st.expander("Attack Metrics"):
              st.dataframe(
                  attack_metrics,
                  use_container_width=True,
                  hide_index=True,
              )