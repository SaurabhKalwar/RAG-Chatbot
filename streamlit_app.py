# streamlit_app.py

import streamlit as st
import requests

st.title("Assistant ChatBot, Your Insurance Companion!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show loading spinner while waiting for response
    with st.spinner("Loading..."):
        try:
            response = requests.post(
                "http://localhost:9003/query",
                json={"question": prompt}
            )
            response.raise_for_status()
            answer = response.json().get("answer", "Sorry, something went wrong.")
        except Exception as e:
            answer = f"Error: {str(e)}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})