import streamlit as st
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os

st.set_page_config(page_title="Bishop's University Academic Calendar Q&A", layout="wide")


def add_fullscreen_background():
    image_path = "image/Purple_Background-scaled.jpg"

    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            image_format = 'jpeg'

            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url(data:image/{image_format};base64,{encoded_string});
                    background-attachment: fixed;
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    margin: 0;
                    padding: 0;
                }}

                .stDeployButton, .stAppHeader, #MainMenu, footer,
                .css-18ni7ap, .css-1rs6os, .css-1v0mbdj, .stToolbar {{
                    display: none !important;
                }}

                .main {{
                    padding: 0;
                    margin: 0;
                }}

                .main .block-container {{
                    padding-top: 1rem;
                    margin-top: 0rem;
                    max-width: 90% !important;  
                    margin-left: auto !important; 
                    margin-right: auto !important; 
                    padding-bottom: 80px !important;
                }}

                .center-content {{
                    text-align: center !important;
                    margin: 1rem 0 !important;
                }}

                .center-title {{
                    text-align: left !important;
                }}

                .column-header {{
                    text-align: left !important;
                    margin-bottom: 1rem !important;
                }}

                .stApp, .stApp > div {{
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }}

                .stApp, .stApp * {{
                    color: white !important;
                }}

                h1, h2, h3, h4, h5, h6 {{
                    color: white !important;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.7) !important;
                }}

                p, div, span, label {{
                    color: white !important;
                }}

                .stRadio > div > label > div {{
                    color: white !important;
                }}

                .stTextInput > label {{
                    color: white !important;
                    font-weight: bold !important;
                }}

                .stTextInput input {{
                    color: black !important;
                    background-color: rgba(255,255,255,0.9) !important;
                }}

                .stTextArea > label {{
                    color: white !important;
                    font-weight: bold !important;
                }}

                .stTextArea textarea {{
                    color: black !important;
                    background-color: rgba(255,255,255,0.9) !important;
                }}

                .stSelectbox > label {{
                    color: white !important;
                }}

                .stAlert {{
                    color: black !important;
                }}

                .stMarkdown {{
                    color: white !important;
                }}

                .matched-section {{
                    background-color: rgba(255,255,255,0.9) !important;
                    color: black !important;
                    padding: 0.5rem 1rem !important;
                    border-radius: 8px !important;
                    margin: 1rem 0 !important;
                    border: 2px solid rgba(255,255,255,0.8) !important;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
                }}

                .matched-section * {{
                    color: black !important;
                }}

                .summary-section {{
                    background-color: rgba(255,255,255,0.9) !important;
                    color: black !important;
                    padding: 1rem !important;
                    border-radius: 8px !important;
                    margin: 1rem 0 !important;
                    border: 2px solid rgba(255,255,255,0.8) !important;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
                }}

                .summary-section * {{
                    color: black !important;
                }}

                .footer {{
                    position: fixed;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                    background-color: rgba(0,0,0,0.7);
                    color: white;
                    text-align: center;
                    padding: 10px 0;
                    font-size: 12px;
                    z-index: 999;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"{str(e)}")
    else:
        st.warning(f"Not found: image/Purple_Background-scaled.jpg")


add_fullscreen_background()

# Main title
st.title("Bishop's University Academic Calendar Assistant")

# Centered answering mode selection
st.markdown('<div class="center-content">', unsafe_allow_html=True)
mode = st.radio(
    "Choose Answering Mode", 
    ["Single Model (GPT-3.5)", "Multi-Model (3.5 + 4)"],
    horizontal=True
)
st.markdown('</div>', unsafe_allow_html=True)

openai_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_key)

SECTION_KEYWORDS = {
    "sessional_dates": ["sessional", "deadline", "date", "calendar", "term begins", "last day", "important dates",
                        "start", "end"],
    "general_information": ["overview", "history", "background", "mission", "general info", "university info"],
    "admission": ["admission", "apply", "application", "requirements", "criteria", "accepted", "how to apply"],
    "fees": ["fee", "tuition", "cost", "pay", "payment", "charge", "billing", "price", "international student fee"],
    "university_regulations": ["regulation", "withdraw", "academic", "rules", "policies", "credit limit",
                               "academic standing", "fail"],
    "programs_courses": ["program", "course", "credits", "major", "minor", "degree", "curriculum", "structure", "hours",
                         "code"],
    "services_facilities": ["residence", "library", "housing", "services", "support", "health", "transport",
                            "student life"],
    "scholarships": ["scholarship", "bursary", "award", "prize", "financial aid", "entrance scholarship", "funding",
                     "grant"],
    "administration": ["dean", "registrar", "president", "principal", "senate", "chancellor", "governance", "trustees"],
    "index": ["index", "reference", "table", "contents"]
}

SECTION_TEXTS = {
    "sessional_dates": """[Page 5] Fall term begins on September 4... [Page 6] Winter term starts on January 6...""",
    "general_information": """[Page 7] Bishop's University was founded in 1843...""",
    "admission": """[Page 9] Students may apply online... Graduate admissions require...""",
    "fees": """[Page 15] Tuition for Canadian students is approximately $7,000...""",
    "university_regulations": """[Page 19] Students must maintain full-time status... Detailed academic policies follow.""",
    "programs_courses": """[Page 37] The BSc in Computer Science requires 90 credits...""",
    "services_facilities": """[Page 253] On-campus housing is available...""",
    "scholarships": """[Page 263] Entrance scholarships are awarded automatically...""",
    "administration": """[Page 287] The university is governed by a Senate...""",
    "index": """[Page 293—299] Index of terms..."""
}


def identify_section(question):
    q = question.lower()
    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', q))
        scores[section] = count
    if all(score == 0 for score in scores.values()):
        return "general_information"
    return max(scores, key=scores.get)


def summarize_with_model(text, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following academic content."},
                {"role": "user", "content": text[:3000]}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"


def compute_similarity_matrix(summaries):
    vect = TfidfVectorizer().fit_transform(summaries)
    return cosine_similarity(vect)


def find_central_summary(matrix, summaries):
    scores = matrix.sum(axis=1)
    return summaries[int(np.argmax(scores))]


def answer_question(question, section_text):
    prompt = f"""You are a helpful academic advisor at Bishop's University.\nUse the following academic calendar section to answer the student's question.\n\nSection:\n{section_text[:3000]}\n\nQuestion:\n{question}\n\nAnswer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {str(e)}"


def summarize_calendar_content(subject, model="gpt-3.5-turbo"):
    """Summarize academic calendar content by subject using specified model"""
    # Find related section
    section = identify_section(subject)
    section_text = SECTION_TEXTS[section]
    
    prompt = f"""You are a helpful academic advisor at Bishop's University. 
Please provide a comprehensive summary about "{subject}" based on the following academic calendar section.
Focus on the key information that would be most useful for students.

Section: {section_text}

Subject: {subject}

Please provide a detailed summary covering all relevant information about this topic:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {str(e)}"


# Create two-column layout
col1, col2 = st.columns([1, 1], gap="large")

# Left: Q&A
with col1:
    st.header("📖 Q&A")

    question = st.text_input(
        "question", 
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

    if question:
        section = identify_section(question)
        st.markdown(f"**Matched Section:** {section.replace('_', ' ').title()}")
        section_text = SECTION_TEXTS[section]

        if mode.startswith("Single"):
            answer = answer_question(question, section_text)
            st.markdown(
                f"""
                <div class="summary-section">
                    <p>{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            models = ["gpt-3.5-turbo", "gpt-4"]
            summaries = [summarize_with_model(section_text, m) for m in models]
            matrix = compute_similarity_matrix(summaries)
            central = find_central_summary(matrix, summaries)

            st.subheader("Similarity Matrix")
            st.dataframe(pd.DataFrame(matrix, index=models, columns=models))

            answer = answer_question(question, central)
            st.markdown(
                f"""
                <div class="summary-section">
                    <p>{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Right: Summarizer
with col2:
    st.header("📃 Summarizer")
    
    subject_input = st.text_input(
        "subject",
        placeholder="e.g., tuition fees, admission requirements, scholarships, deadlines...",
        label_visibility="collapsed"
    )
    
    # Auto-generate summary when input provided
    if subject_input.strip():
        with st.spinner("Generating summary..."):
            section = identify_section(subject_input)
            
            if mode.startswith("Single"):
                summary = summarize_calendar_content(subject_input, "gpt-3.5-turbo")
                st.markdown(f"**Related Section:** {section.replace('_', ' ').title()}")
                st.markdown(
                    f"""
                    <div class="summary-section">
                        <p>{summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Multi-model approach for summarizer
                models = ["gpt-3.5-turbo", "gpt-4"]
                summaries = [summarize_calendar_content(subject_input, model) for model in models]
                matrix = compute_similarity_matrix(summaries)
                central_summary = find_central_summary(matrix, summaries)
                
                st.markdown(f"**Related Section:** {section.replace('_', ' ').title()}")
                st.subheader("Similarity Matrix")
                st.dataframe(pd.DataFrame(matrix, index=models, columns=models))
                
                st.markdown(
                    f"""
                    <div class="summary-section">
                        <p>{central_summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Footer
st.markdown(
    """
    <div class="footer">
        Copyright © BU. Version 0.1. Last update August 2025
    </div>
    """,
    unsafe_allow_html=True
)

