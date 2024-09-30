from pdf_operation import load_pdf, generate_embeddings
from utils import to_base64, create_download_link
from client import create_client, embedding_text, completion_text
import openai
import os
import uuid
import streamlit as st
import re
from dotenv import load_dotenv
import numpy as np

load_dotenv()
print(f"loaded API_KEY: {os.getenv('API_KEY')}")
print(f"loaded API_BASE: {os.getenv('API_BASE')}")
print(f"loaded EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")
print(f"loaded COMPLETION_MODEL: {os.getenv('COMPLETION_MODEL')}")

# Define the default state of the app
DEFAULT_STATE = {
    'files': [],
    'last_uploaded_file_name': "",
    "run_log": "",
    "chat_history": [
        {"role": "system", "content": "You are a helpful assistant."},
    ],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

client = create_client()
# embedding_text(client, "Hello, how are you?")
# response = completion_text(
#     client,
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print(response)


def main():

    # title
    st.title('RAG-enhanced PDF Chatbot')

    # write a line jsonifying the session state
    # st.write(st.session_state)

    first_col, second_col = st.columns(2)

    with first_col:
        st.header('Upload PDFs Here')

        # limit to 20 mb
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type='pdf',
            key='file_uploader',
            accept_multiple_files=False)

        if (uploaded_file is not None and uploaded_file.name != st.session_state.get('last_uploaded_file_name', None)):
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'deleted': False,  # setting the 'deleted' key to False
                'obj_storage_location': os.path.join('uploaded_files', str(uuid.uuid4())),
                'processing_status': 0,  # -1 means done, 0 means not started, 1, 2 etc means each step
                'embeddings': []
                # Example of embeddings format
                # [
                #     {
                #         "text": page_content,
                #         "embedding": embedding_text(openai_client, page_content)
                #     }
                # ]
            }

            # save it at obj_storage_location
            # first check folder exists
            if not os.path.exists('uploaded_files'):
                os.makedirs('uploaded_files')
            with open(file_info['obj_storage_location'], 'wb') as f:
                f.write(uploaded_file.read())

            st.session_state['files'] = st.session_state.get(
                'files', []).copy() + [file_info]
            st.session_state['last_uploaded_file_name'] = uploaded_file.name
            st.write('File uploaded successfully!')
            st.session_state['run_log'] = st.session_state['run_log'] + \
                f"Uploaded file: {uploaded_file.name}"

            # ---------------------------------- Section sep ------------------------------------
            __output = load_pdf(file_info['obj_storage_location'])
            # change the streamlit
            __output = generate_embeddings(__output, client)
            st.session_state['embeddings'] = __output
            # ---------------------------------- Section sep ------------------------------------
            st.rerun()

    with second_col:
        st.header('Uploaded Files')

        if 'files' in st.session_state:
            for i, uploaded_file in reversed(list(enumerate(st.session_state['files']))):
                # Only render files that are not deleted
                if not uploaded_file['deleted']:

                    col1, col2, col3 = st.columns([5, 1, 1])
                    # center vertically

                    with col1:
                        st.write(uploaded_file['name'])

                    with col2:
                        if st.button(f"⬇️", key=f'prepare_{i}'):
                            st.markdown(create_download_link(
                                uploaded_file['obj_storage_location'], uploaded_file['name']), unsafe_allow_html=True)

                    with col3:
                        if st.button(f"🗑️", key=f'delete_{i}'):
                            new_file_info = uploaded_file.copy()  # Make a copy of the dictionary
                            new_file_info['deleted'] = True
                            st.session_state['files'][i] = new_file_info
                            st.rerun()

    # first title
    st.header('Chat with the Bot')
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            st.markdown(
                f'<p style="color:rgb(243, 142, 103);">User:</p>', unsafe_allow_html=True)
            for line in chat["content"].split("\n"):
                st.markdown(
                    f'<p style="color:rgb(243, 142, 103);">{line}</p>', unsafe_allow_html=True)

        else:
            st.markdown(
                f'<p style="color:rgb(177, 241, 101);">Chatbot:</p>', unsafe_allow_html=True)
            for line in chat["content"].split("\n"):
                st.markdown(
                    f'<p style="color:rgb(177, 241, 101);">{line}</p>', unsafe_allow_html=True)

    user_input = st.text_area("Type your message here", key='user_input')
    if st.button("Send", key='send_button'):
        user_input = re.sub(r'\n+', '\n', user_input).strip()
        st.session_state['chat_history'] = st.session_state.get(
            'chat_history', []).copy() + [{'role': 'user', 'content': user_input}]

        messages = []
        for chat in st.session_state['chat_history']:
            if chat['role'] == 'user':
                messages.append({"role": "user", "content": chat["content"]})
            elif chat['role'] == 'system':
                messages.append({"role": "system", "content": chat["content"]})

        user_input_embedding = embedding_text(client, user_input)

        similarity = []

        for embedding in st.session_state['embeddings']:
            print(embedding, flush=True)
            similarity.append(
                (embedding['text'], np.dot(
                    user_input_embedding, embedding['embedding']))
            )

        similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

        # get top 3
        similarity = similarity[:3]

        # only sent the last message, then append these 3 top similarity text
        sent_message = [DEFAULT_STATE['chat_history'][0]]
        sent_message.append(
            {
                "role": "user",
                "content": user_input +
                "\n" +
                "Here are some context that might be helpful:" +
                "\n" +
                "\n".join([f"{i+1}. {sim[0]}" for i,
                          sim in enumerate(similarity)]) +
                user_input
            }
        )

        respond_message = completion_text(client, sent_message)
        respond_message = re.sub(r'\n+', '\n', respond_message).strip()

        st.session_state['chat_history'] = st.session_state.get(
            'chat_history', []).copy() + [{'role': 'system', 'content': respond_message}]

        st.rerun()


if __name__ == '__main__':
    main()
