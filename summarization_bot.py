import os
import openai
import streamlit as st
import io
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
import PyPDF2

# [Previous functions remain exactly the same]
def transcribe_audio(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        buffer.name = "audio.wav"
        response = openai.Audio.transcribe(
            "whisper-1",
            file=buffer,
            response_format="verbose_json"
        )
        return response
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Split text into chunks of approximately 1500 words each
    words = text.split()
    chunks = [" ".join(words[i:i + 1500]) for i in range(0, len(words), 1500)]
    return chunks

def get_youtube_transcript(url):
    try:
        # Extract video ID from URL
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            st.error("Invalid YouTube URL.")
            return None
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Choose a transcript
        transcript = transcript_list.find_transcript(['en'])
        # Fetch the actual transcript data
        transcript_data = transcript.fetch()
        # Combine transcript texts
        transcription_text = "\n".join([f"[{entry['start']:.2f}-{entry['start']+entry['duration']:.2f}] {entry['text']}" for entry in transcript_data])
        return transcription_text
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {str(e)}")
        return None

def generate_summary(text):
    prompt = f"Summarize the following text into 500-800 words, focusing on key points:\n\n{text}"
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()

def chat_with_ai(question, context):
    prompt = f"Based on the following content:\n\n{context}\n\nAnswer the question:\n{question}"
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def main():
    st.set_page_config(layout="wide")

    # Add custom CSS for gradient background
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(180deg, 
                rgba(64,224,208,0.7) 0%, 
                rgba(32,112,104,0.4) 35%, 
                rgba(0,0,0,0) 100%
            );
        }
        .css-1d391kg {
            background: none;
        }
        .stMarkdown {
            color: #ffffff;
        }
        .css-1y4p8pa {
            max-width: 100%;
            padding: 2rem;
        }
        div[data-testid="stSidebarContent"] {
            background-color: rgba(255,255,255,0.1);
        }
        .stTextArea textarea {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .stButton button {
            background-color: #40E0D0;
            color: black;
        }
        .stButton button:hover {
            background-color: #48D1CC;
            color: black;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        .css-184tjsw p {
            color: white !important;
        }
        .stTextInput input {
            color: white !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>AI Summarization Bot 🤖</h1>", unsafe_allow_html=True)

    # Left sidebar for upload options
    st.sidebar.header("Upload your file:")
    input_type = st.sidebar.selectbox("Select Input Type", ["Audio File", "PDF Document", "YouTube URL"])
    st.sidebar.markdown("### Steps to Use the Tool:")
    st.sidebar.markdown("1. Upload an audio file (25MB max), PDF, or YouTube URL for summary.")
    st.sidebar.markdown("2. Click on 'Generate Summary' to get an AI-transcribed summary.")
    st.sidebar.markdown("3. Use 'Chat with AI' for further questions.")

    # File uploader or URL input based on selected type
    audio_input = pdf_input = youtube_input = None
    if input_type == "Audio File":
        audio_input = st.sidebar.file_uploader("Upload audio file", type=["mp3", "wav"], key="audio", help="Supports mp3 and wav formats up to 25MB")
        if st.sidebar.button("Generate Summary", key="generate_summary_button"):
            st.session_state['generate_summary_clicked'] = True
    elif input_type == "PDF Document":
        pdf_input = st.sidebar.file_uploader("Upload PDF document", type=["pdf"], key="pdf", help="")
    elif input_type == "YouTube URL":
        youtube_input = st.sidebar.text_input("Enter YouTube URL (must have subtitles enabled)", key="youtube")

    # Main section for processing and results
    if 'generate_summary_clicked' in st.session_state and st.session_state['generate_summary_clicked']:
        transcription_text = ""
        if input_type == "Audio File" and audio_input:
            transcription = transcribe_audio(audio_input)
            if transcription:
                transcription_text = "\n".join(
                    [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in transcription['segments']]
                )
            else:
                st.error("Transcription failed.")
        elif input_type == "PDF Document" and pdf_input:
            chunks = extract_text_from_pdf(pdf_input)
            summaries = [generate_summary(chunk) for chunk in chunks]
            transcription_text = "\n".join(summaries)
        elif input_type == "YouTube URL" and youtube_input:
            transcription_text = get_youtube_transcript(youtube_input)
            if not transcription_text:
                st.error("Failed to retrieve YouTube transcript.")
        else:
            st.error("Please provide valid input.")

        if transcription_text:
            st.session_state['transcription'] = transcription_text
            summary = generate_summary(transcription_text)
            st.session_state['summary'] = summary

    # Display transcription and summary side by side
    if 'transcription' in st.session_state and 'summary' in st.session_state:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<h3>Transcription with Timestamps</h3>", unsafe_allow_html=True)
            st.text_area("", st.session_state['transcription'], height=300)

        with col2:
            st.markdown("<h3>Summary</h3>", unsafe_allow_html=True)
            st.text_area("", st.session_state['summary'], height=300)

        # Chat with AI at the bottom
        st.markdown("---")
        st.markdown("<h3>Chat with AI</h3>", unsafe_allow_html=True)
        question = st.text_input("Ask a question about the content:")
        if st.button("Ask AI"):
            if 'summary' in st.session_state:
                answer = chat_with_ai(question, st.session_state['summary'])
                st.markdown("<p style='color: white; font-weight: bold;'>AI Answer:</p>", unsafe_allow_html=True)
                st.write(answer)
            else:
                st.error("Please generate a summary first.")
