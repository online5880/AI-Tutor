import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# env 불러오기
load_dotenv()

# model 생성 - OpenAI
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

chain = model | StrOutputParser()

# 대화 기록 세션 저장
if "messages" not in st.session_state:
    st.session_state.messages = []
    
st.title("대화 테스트")

# openai api 와 형태 맞추기 - role, content
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
user_input = st.chat_input("입력")

if user_input:
    st.session_state.messages.append({'role':'user','content':user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_token = ""
        for token in chain.stream(user_input):
            full_token+= token
            message_placeholder.markdown(full_token + "|")
        message_placeholder.markdown(full_token)
    st.session_state.messages.append({"role":"assistant","content":full_token})