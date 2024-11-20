import os
import io
import openai
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging
import warnings
from pydub import AudioSegment
import tempfile
import PyPDF2
from typing import List, Dict, Optional, IO, Union
from pathlib import Path
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 25
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav"]
SUPPORTED_PDF_FORMAT = ".pdf"
MAX_TOKENS = 1500
API_CALLS_PER_MINUTE = 50

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class FileProcessingError(Exception):
    """Custom exception for file processing errors."""
    pass

@sleep_and_retry
@limits(calls=API_CALLS_PER_MINUTE, period=60)
def rate_limited_api_call(func):
    """Decorator for rate-limiting API calls."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def validate_api_key() -> bool:
    """Validate that the OpenAI API key is set and valid."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError("OpenAI API key not found in environment variables")
    openai.api_key = api_key
    return True

def validate_file_size(file: IO[bytes], max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """Validate that the file size is within limits."""
    file.seek(0, 2)  # Seek to end of file
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    return file_size <= max_size_mb * 1024 * 1024

def extract_text_from_pdf(file: IO[bytes]) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise FileProcessingError(f"Error processing PDF: {str(e)}")

def convert_to_supported_format(file: IO[bytes]) -> IO[bytes]:
    """Convert audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(file)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise FileProcessingError(f"Error converting audio: {str(e)}")

def safe_temp_file_cleanup(temp_file_path: str) -> None:
    """Safely clean up temporary files."""
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except OSError as e:
        logger.error(f"Error cleaning up temp file: {e}")

@rate_limited_api_call
def transcribe_audio(file: IO[bytes]) -> str:
    """Transcribe audio file using OpenAI's Whisper API."""
    if not validate_file_size(file):
        raise FileProcessingError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")

    file = convert_to_supported_format(file)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        raise FileProcessingError(f"Error in transcription: {str(e)}")
    finally:
        safe_temp_file_cleanup(temp_file_path)

@rate_limited_api_call
def summarize_text(text: str) -> str:
    """Generate a summary of the text using OpenAI's API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise FileProcessingError(f"Error in summarization: {str(e)}")

@rate_limited_api_call
def generate_quiz_questions(text: str) -> str:
    """Generate quiz questions using OpenAI's API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates quiz questions. Your task is to generate ten quiz questions and four multiple choice answers for each question from the given text. It is CRUCIAL that you mark the correct answer with an asterisk (*) at the beginning of the answer line. There MUST be exactly one correct answer marked for each question."},
                {"role": "user", "content": f"Generate quiz questions from the following text:\n\n{text}"}
            ],
            max_tokens=MAX_TOKENS
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise FileProcessingError(f"Error in quiz generation: {str(e)}")

@rate_limited_api_call
def generate_explanation(question: str, correct_answer: str, user_answer: str) -> str:
    """Generate explanation for incorrect answers."""
    try:
        prompt = f"Explain why the correct answer to the following question is '{correct_answer}' and not '{user_answer}':\n\n{question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise FileProcessingError(f"Error in explanation generation: {str(e)}")

def get_transcript(url: str) -> str:
    """Get transcript from YouTube video."""
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            raise ValueError("Invalid YouTube URL")
        
        video_id = video_id_match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        raise FileProcessingError(f"Error getting YouTube transcript: {str(e)}")

def parse_quiz_questions(quiz_text: str) -> List[Dict[str, Union[str, List[str]]]]:
    """Parse quiz questions from generated text."""
    questions = []
    question_blocks = quiz_text.split("\n\n")
    
    for block in question_blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 5:
            question = lines[0].split(". ", 1)[1] if ". " in lines[0] else lines[0]
            choices = []
            correct_answer = None
            
            for line in lines[1:5]:
                if ") " in line:
                    choice = line.split(") ", 1)[1].strip()
                    if choice.startswith("*"):
                        correct_answer = choice[1:].strip()
                        choices.append(correct_answer)
                    else:
                        choices.append(choice)
            
            if choices:
                questions.append({
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_answer or choices[0]
                })
    
    return questions

def check_answers(questions: List[Dict], user_answers: Dict) -> List[Dict]:
    """Check user answers and generate feedback."""
    feedback = []
    for i, question in enumerate(questions):
        correct_answer = question['correct_answer']
        user_answer = user_answers.get(f"question_{i+1}", "")

        feedback_item = {
            "question": question['question'],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "status": "Correct" if user_answer == correct_answer else "Incorrect"
        }

        if user_answer != correct_answer:
            feedback_item["explanation"] = generate_explanation(
                question['question'],
                correct_answer,
                user_answer
            )

        feedback.append(feedback_item)

    return feedback


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""


def main():
    """Main application function."""
    try:
        validate_api_key()
    except ConfigurationError as e:
        st.error(str(e))
        return

    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: #40E0D0;'>AI Quiz Generator 🤖</h1>", unsafe_allow_html=True)

    # Add custom CSS for turquoise UI
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
        div[data-testid="stSidebarContent"] {
            background-color: rgba(255,255,255,0.1);
        }
        .stTextArea textarea, .stTextInput input {
            background-color: rgba(0, 0, 0, 0.5) !important;
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
            color: #ffffff !important;
        }
        .css-184tjsw p {
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar
    st.sidebar.header("Upload your file:")
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ["Audio File", "PDF Document", "YouTube URL"]
    )

    st.sidebar.markdown("### Steps:")
    st.sidebar.markdown("1. Upload a file or provide a YouTube URL")
    st.sidebar.markdown("2. Click 'Generate Quiz'")
    st.sidebar.markdown("3. Answer questions and submit")

    try:
        if input_type == "Audio File":
            handle_audio_input()
        elif input_type == "PDF Document":
            handle_pdf_input()
        elif input_type == "YouTube URL":
            handle_youtube_input()

        if st.session_state.questions:
            display_quiz()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)


def handle_audio_input():
    """Handle audio file input."""
    audio_input = st.sidebar.file_uploader("Upload audio file", type=["mp3", "wav"])
    if audio_input and st.sidebar.button("Generate Quiz"):
        with st.spinner('Transcribing audio...'):
            transcription_text = transcribe_audio(audio_input)
            process_text_input(transcription_text)


def handle_pdf_input():
    """Handle PDF file input."""
    pdf_input = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
    if pdf_input and st.sidebar.button("Generate Quiz"):
        with st.spinner('Reading PDF...'):
            text = extract_text_from_pdf(pdf_input)
            process_text_input(text)


def handle_youtube_input():
    """Handle YouTube URL input."""
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url and st.sidebar.button("Generate Quiz"):
        with st.spinner('Fetching transcript...'):
            transcript_text = get_transcript(youtube_url)
            process_text_input(transcript_text)


def process_text_input(text: str):
    """Process input text and generate quiz."""
    st.session_state.transcript_text = text
    quiz_text = generate_quiz_questions(text)
    st.session_state.questions = parse_quiz_questions(quiz_text)
    st.session_state.user_answers = {}
    st.session_state.feedback = []


def display_quiz():
    """Display quiz questions and handle submissions."""
    st.markdown("## Quiz Questions")
    form = st.form(key="quiz_form")

    for i, question in enumerate(st.session_state.questions):
        form.markdown(f"**Question {i+1}: {question['question']}**")
        user_answer = form.radio(
            f"Choose your answer for Question {i+1}:",
            options=question["choices"],
            key=f"question_{i+1}"
        )
        st.session_state.user_answers[f"question_{i+1}"] = user_answer

    if form.form_submit_button("Submit Answers"):
        with st.spinner("Checking answers..."):
            st.session_state.feedback = check_answers(
                st.session_state.questions,
                st.session_state.user_answers
            )
        display_feedback()


def display_feedback():
    """Display quiz feedback."""
    st.markdown("<h2 style='text-align: center; color: #40E0D0;'>Quiz Results</h2>", unsafe_allow_html=True)

    correct_count = sum(1 for f in st.session_state.feedback if f["status"] == "Correct")
    total_questions = len(st.session_state.feedback)

    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 30px; color: #ffffff;'>
            <h3>Your Score: {correct_count}/{total_questions}</h3>
            <p>({(correct_count/total_questions * 100):.1f}%)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    for i, feedback in enumerate(st.session_state.feedback, 1):
        status_color = "#28a745" if feedback["status"] == "Correct" else "#dc3545"
        st.markdown(
            f"""
            <div style='padding: 15px; border-left: 5px solid {status_color}; color: #ffffff;'>
                <h4>Question {i}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"**{feedback['question']}**")
        user_answer_color = "green" if feedback["status"] == "Correct" else "red"
        st.markdown(f"Your Answer: <span style='color: {user_answer_color};'>{feedback['user_answer']}</span>",
                    unsafe_allow_html=True)

        if feedback["status"] == "Incorrect":
            st.markdown(f"Correct Answer: <span style='color: green;'>{feedback['correct_answer']}</span>",
                        unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; color: #000000;'>
                    <strong>Explanation:</strong><br>
                    {feedback['explanation']}
                </div>
                """,
                unsafe_allow_html=True
            )

        if i < total_questions:
            st.markdown(
                """
                <hr style='margin: 20px 0; border: none; height: 1px; background-color: #e0e0e0;'>
                """,
                unsafe_allow_html=True
            )

    if st.button("Try Another Quiz"):
        st.session_state.questions = []
        st.session_state.user_answers = {}
        st.session_state.feedback = []
        st.session_state.transcript_text = ""
        st.experimental_rerun()


if __name__ == "__main__":
    main()
