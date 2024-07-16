import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Change: Use os.getenv for better error handling
groq_api_key = os.getenv('API_KEY')

def main():
    st.title("Your Personal LLM")
    st.sidebar.title("Select the LLM model")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['Mixtral-8x7b-32768',
         #'llama2-70b-4096']
    ])
    conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)
    
    memory = ConversationBufferMemory(k=conversational_memory_length)
    
    user_question = st.text_area('ASK a question...')
    
    # Session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Empty list given here
    else:
        # Change: Consistent key names
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})
    
    # Initiating a Groq chat session
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )
    
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )
    
    # Only run the model if user asked a question and hit enter
    if user_question:
        response = conversation.invoke(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        st.write("chatbot:", response['response'])

# Change: Correct indentation
if __name__ == '__main__':
    main()
