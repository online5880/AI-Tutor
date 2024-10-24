import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import base64
import whisper
from io import BytesIO
from gtts import gTTS

# env 불러오기
load_dotenv()

# model 생성 - OpenAI
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

model = OpenAI()

whisper_model = whisper.load_model("tiny")

chain = chat_model | StrOutputParser()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Helpers
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

# View
st.title("음성 대화 테스트")

con1 = st.container()
con2 = st.container()

user_input = ""

with con2:
    audio_bytes = audio_recorder("talk", pause_threshold=3.0,)
    try:
        if audio_bytes:
            with open("./tmp_audio.wav", "wb") as f:
                f.write(audio_bytes)

            with open("./tmp_audio.wav", "rb") as f: 
                print("start transcript")
                print(f)
                transcript = whisper_model.transcribe("./tmp_audio.wav")
                user_input = transcript['text']
                print(user_input)
    except Exception as e:
        pass


with con1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_token = ""
            for token in chain.stream(user_input):
                full_token+= token
                message_placeholder.markdown(full_token + "|")
            message_placeholder.markdown(full_token)

            speech_file_path = "tmp_speak.mp3"
            # response = model.audio.speech.create(
            #   model="tts-1",
            #   voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
            #   input=full_token
            # )
            # response.stream_to_file(speech_file_path)
            tts = gTTS(text=full_token, lang="ko")
            tts.save(speech_file_path)
            autoplay_audio(speech_file_path)

        st.session_state.messages.append({"role": "assistant", "content": full_token})


