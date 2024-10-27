import base64
import streamlit as st
from openai import OpenAI
import yaml
from audio_recorder_streamlit import audio_recorder
from langchain.prompts import load_prompt
import os  # 임시 파일 삭제를 위해 추가

# 상수
TMP_AUDIO_PATH = "./tmp_audio.wav"
TMP_SPEECH_PATH = "tmp_speak.mp3"

# OpenAI 클라이언트 초기화
client = OpenAI()

# YAML 파일에서 프롬프트 로드
def load_prompts_from_yaml(file_path):
    with open(file_path, "r") as file:
        prompts_data = yaml.safe_load(file)
    return {
        "초급": prompts_data["beginner"]["template"],
        "중급": prompts_data["intermediate"]["template"],
        "고급": prompts_data["advanced"]["template"],
    }

# 레벨 프롬프트 로드
level_prompts = load_prompts_from_yaml("prompt/prompts.yaml")

# 세션 상태 초기화
if "level" not in st.session_state:
    st.session_state.level = "초급"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": level_prompts[st.session_state.level].format()}]

if "prev_audio_bytes" not in st.session_state:
    st.session_state.prev_audio_bytes = None

# 오디오를 자동 재생으로 표시하는 헬퍼 함수
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# 정책 위반 체크 헬퍼 함수
def get_policy_violated(text):
    response = client.moderations.create(input=text)
    output = response.results[0]
    flagged_list = [(k, v) for k, v in output.model_dump()['categories'].items() if v]
    return flagged_list

# UI 구성 요소
st.title("프리 토킹 테스트")

# 새로운 대화 버튼
if st.button("새로운 대화"):
    st.session_state.messages = [{"role": "system", "content": level_prompts[st.session_state.level].format()}]

# 난이도 선택 및 프롬프트 업데이트
level = st.selectbox("난이도", ["초급", "중급", "고급"])
if level != st.session_state.level:
    st.session_state.level = level
    st.session_state.messages = [{"role": "system", "content": level_prompts[st.session_state.level].format()}]

# 오디오 녹음
audio_bytes = audio_recorder("말하기", pause_threshold=3.0)
if audio_bytes != st.session_state.prev_audio_bytes:
    st.session_state.prev_audio_bytes = audio_bytes
else:
    audio_bytes = None

# 사용자의 오디오 입력 처리
user_input = ""
if audio_bytes:
    try:
        with open(TMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        # 오디오를 텍스트로 변환
        with open(TMP_AUDIO_PATH, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
            user_input = transcript.text

    except Exception as e:
        st.error(f"오디오 처리 오류: {e}")
    finally:
        # 임시 오디오 파일 삭제
        os.remove(TMP_AUDIO_PATH)

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 및 어시스턴트 응답 처리
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # 사용자 입력에 대한 정책 위반 체크
    flagged_content = get_policy_violated(user_input)
    if flagged_content:
        st.warning(flagged_content)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # OpenAI의 챗 완성에서 스트리밍 응답 생성
        for response in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 어시스턴트의 오디오 응답 생성 및 재생
        speech_response = client.audio.speech.create(model="tts-1", voice="alloy", input=full_response)
        speech_response.stream_to_file(TMP_SPEECH_PATH)
        autoplay_audio(TMP_SPEECH_PATH)

        # 생성된 음성 파일 삭제
        os.remove(TMP_SPEECH_PATH)
