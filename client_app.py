import streamlit as st
from src.agents import run_agent

st.set_page_config(
    page_title="FoodBundles AI Assistant",
    page_icon="🍱",
    layout="centered",
)

# --- Header ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/meal.png", width=56)
with col2:
    st.title("FoodBundles AI Assistant")
    st.caption("Your smart guide to ordering, payments, and more on [food.rw](https://food.rw)")

st.divider()

# --- Chat History ---
if "client_messages" not in st.session_state:
    st.session_state.client_messages = []

for msg in st.session_state.client_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask anything about FoodBundles..."):
    st.session_state.client_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(prompt, st.session_state.client_messages)
        st.markdown(response)

    st.session_state.client_messages.append({"role": "assistant", "content": response})

# --- Footer ---
st.divider()
st.caption("🔒 This assistant never shares sensitive or internal data. | Powered by [FoodBundles](https://food.rw)")
