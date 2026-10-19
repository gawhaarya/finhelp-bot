import streamlit as st
from src.rag_pipeline import answer_question

st.set_page_config(
    page_title="FinHelp Bot",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 FinHelp Bot")
st.caption("Ask questions about the available banking policy documents.")

st.info(
    "This assistant answers using the provided training policy. "
    "It is a demo, not an official bank service."
)

# Keep conversation history across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display earlier messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show source information when available.
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"{source['source']} — "
                        f"page {source['page_number']}"
                    )

# Accept a new question.
question = st.chat_input("Ask a banking policy question...")

# if question:
#     # Show and save the user's question.
#     st.session_state.messages.append({
#         "role": "user",
#         "content": question,
#     })

#     with st.chat_message("user"):
#         st.markdown(question)

#     # Retrieve context and generate the answer.
#     with st.chat_message("assistant"):
#         with st.spinner("Checking the policy documents..."):
#             try:
#                 answer = answer_question(question)
#                 st.markdown(answer)

#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": answer,
#                 })

#             except Exception as error:
#                 st.error(
#                     "Sorry, something went wrong while answering "
#                     "your question. Check the terminal for details."
#                 )
#                 print(f"FinHelp Bot error: {error}")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the policy documents..."):
            try:
                result = answer_question(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(
                                f"{source['source']} — "
                                f"page {source['page_number']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(
                    "Sorry, something went wrong while answering "
                    "your question. Check the terminal for details."
                )
                print(f"FinHelp Bot error: {error}")