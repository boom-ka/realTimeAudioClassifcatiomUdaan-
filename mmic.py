import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np

class AudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        st.session_state['last_audio'] = audio
        return frame

st.title("🎙️ Microphone Test")

webrtc_streamer(
    key="test-stream",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_processor_factory=AudioProcessor,
)

if 'last_audio' in st.session_state:
    st.line_chart(st.session_state['last_audio'][:512])
