import streamlit as st
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os
import PyPDF2
import io

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

                .pdf-status {{
                    background-color: rgba(0, 255, 0, 0.2);
                    color: white;
                    text-align: center;
                    padding: 8px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    border: 1px solid rgba(255,255,255,0.3);
                }}

                .pdf-error {{
                    background-color: rgba(255, 0, 0, 0.2);
                    color: white;
                    text-align: center;
                    padding: 8px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    border: 1px solid rgba(255,0,0,0.5);
                }}

                .section-indicator {{
                    background-color: rgba(0, 191, 255, 0.3);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                    display: inline-block;
                    margin-bottom: 10px;
                    border: 1px solid rgba(255,255,255,0.4);
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"{str(e)}")
    else:
        st.warning(f"Not found: image/Purple_Background-scaled.jpg")


# Table of Contents structure based on the provided catalog
TABLE_OF_CONTENTS = {
    "sessional_dates": {
        "title": "Sessional Dates",
        "start_page": 5,
        "end_page": 6,
        "keywords": ["sessional", "deadline", "date", "calendar", "term begins", "last day", "important dates",
                     "start", "end", "semester", "exam", "registration", "fall", "winter", "spring"]
    },
    "general_information": {
        "title": "General Information", 
        "start_page": 7,
        "end_page": 8,
        "keywords": ["overview", "history", "background", "mission", "general info", "university info", "about",
                     "founded", "campus", "location"]
    },
    "admission": {
        "title": "Admission",
        "start_page": 9,
        "end_page": 14,
        "keywords": ["admission", "apply", "application", "requirements", "criteria", "accepted", "how to apply", 
                     "entrance", "eligibility", "prerequisite", "grade", "average"]
    },
    "fees": {
        "title": "Fees",
        "start_page": 15,
        "end_page": 18,
        "keywords": ["fee", "tuition", "cost", "pay", "payment", "charge", "billing", "price", 
                     "international student fee", "financial", "money", "expense"]
    },
    "university_regulations": {
        "title": "University Regulations",
        "start_page": 19,
        "end_page": 42,
        "keywords": ["regulation", "withdraw", "academic", "rules", "policies", "credit limit",
                     "academic standing", "fail", "probation", "suspension", "policy", "procedure"]
    },
    "programs_courses": {
        "title": "Programs and Courses",
        "start_page": 43,
        "end_page": 266,
        "keywords": ["program", "course", "credits", "major", "minor", "degree", "curriculum", "structure", "hours",
                     "code", "bachelor", "master", "prerequisite", "faculty", "department"]
    },
    "business": {
        "title": "Williams School of Business",
        "start_page": 47,
        "end_page": 72,
        "keywords": ["business", "commerce", "management", "accounting", "finance", "marketing", "economics",
                     "williams", "bba", "entrepreneurship"]
    },
    "education": {
        "title": "School of Education",
        "start_page": 73,
        "end_page": 90,
        "keywords": ["education", "teaching", "teacher", "pedagogy", "b.ed", "bed", "classroom", "learning"]
    },
    "humanities": {
        "title": "Faculty of Humanities",
        "start_page": 91,
        "end_page": 168,
        "keywords": ["humanities", "arts", "english", "history", "philosophy", "drama", "music", "language",
                     "literature", "culture", "fine arts", "classical"]
    },
    "sciences": {
        "title": "Faculty of Natural Sciences and Mathematics",
        "start_page": 169,
        "end_page": 214,
        "keywords": ["science", "math", "mathematics", "biology", "chemistry", "physics", "computer science",
                     "astronomy", "biochemistry", "natural", "pre-medicine"]
    },
    "social_sciences": {
        "title": "Faculty of Social Sciences",
        "start_page": 215,
        "end_page": 266,
        "keywords": ["social science", "psychology", "sociology", "politics", "geography", "environment",
                     "economics", "sports", "international studies"]
    },
    "graduate_programs": {
        "title": "Graduate Programs",
        "start_page": 267,
        "end_page": 294,
        "keywords": ["graduate", "master", "masters", "m.ed", "med", "m.a", "ma", "phd", "doctorate",
                     "postgraduate", "thesis", "research"]
    },
    "services_facilities": {
        "title": "Services and Facilities",
        "start_page": 295,
        "end_page": 304,
        "keywords": ["residence", "library", "housing", "services", "support", "health", "transport",
                     "student life", "facilities", "dining", "recreation", "campus"]
    },
    "scholarships": {
        "title": "Scholarships, Awards, Bursaries, Loans, and Prizes",
        "start_page": 305,
        "end_page": 328,
        "keywords": ["scholarship", "bursary", "award", "prize", "financial aid", "entrance scholarship", "funding",
                     "grant", "money", "loan", "assistance"]
    },
    "administration": {
        "title": "Administration and Librarians",
        "start_page": 329,
        "end_page": 340,
        "keywords": ["dean", "registrar", "president", "principal", "senate", "chancellor", "governance", "trustees",
                     "faculty", "staff", "administration", "librarian"]
    }
}


@st.cache_data
def load_pdf_content():
    """Load and parse the PDF file content with table of contents structure"""
    pdf_file_path = "BU-AcCal-2025-2026-REV-July-2.pdf"
    
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Store page contents
            page_contents = {}
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                page_contents[page_num + 1] = page_text
            
            # Extract sections based on table of contents
            sections = {}
            for section_key, section_info in TABLE_OF_CONTENTS.items():
                start_page = section_info["start_page"]
                end_page = min(section_info["end_page"], total_pages)
                
                section_text = ""
                for page_num in range(start_page, end_page + 1):
                    if page_num in page_contents:
                        section_text += f"\n[Page {page_num}]\n{page_contents[page_num]}\n"
                
                sections[section_key] = {
                    "title": section_info["title"],
                    "content": section_text,
                    "page_range": f"{start_page}-{end_page}",
                    "keywords": section_info["keywords"]
                }
            
            return {
                "sections": sections,
                "page_contents": page_contents,
                "total_pages": total_pages,
                "status": "success"
            }
    
    except FileNotFoundError:
        return {
            "sections": {},
            "page_contents": {},
            "total_pages": 0,
            "status": "file_not_found",
            "error": f"PDF file '{pdf_file_path}' not found. Please ensure the file is in the same directory as this script."
        }
    except Exception as e:
        return {
            "sections": {},
            "page_contents": {},
            "total_pages": 0,
            "status": "error",
            "error": f"Error reading PDF: {str(e)}"
        }


def identify_relevant_sections(question, top_k=2):
    """Identify the most relevant sections based on keywords and content similarity"""
    question_lower = question.lower()
    section_scores = {}
    
    # Calculate keyword matching scores
    for section_key, section_info in TABLE_OF_CONTENTS.items():
        keyword_score = 0
        for keyword in section_info["keywords"]:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', question_lower):
                keyword_score += 1
        section_scores[section_key] = keyword_score
    
    # If no keyword matches, use TF-IDF similarity if PDF is loaded
    if all(score == 0 for score in section_scores.values()) and pdf_content["status"] == "success":
        try:
            # Create documents for similarity calculation
            documents = [question]
            section_keys = []
            
            for section_key, section_data in pdf_content["sections"].items():
                if section_data["content"].strip():
                    documents.append(section_data["content"][:1000])  # First 1000 chars
                    section_keys.append(section_key)
            
            if len(documents) > 1:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Calculate similarity between question and sections
                question_vector = tfidf_matrix[0:1]
                section_vectors = tfidf_matrix[1:]
                similarities = cosine_similarity(question_vector, section_vectors).flatten()
                
                # Update scores with similarity
                for i, section_key in enumerate(section_keys):
                    section_scores[section_key] = similarities[i]
        
        except Exception as e:
            # Fallback to general_information if similarity calculation fails
            section_scores["general_information"] = 1.0
    
    # Get top sections
    if all(score == 0 for score in section_scores.values()):
        return ["general_information"]  # Default fallback
    
    # Sort by score and return top k
    sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
    relevant_sections = [section for section, score in sorted_sections[:top_k] if score > 0]
    
    return relevant_sections if relevant_sections else ["general_information"]


def get_section_content(section_keys):
    """Get content for specified sections"""
    if pdf_content["status"] != "success":
        return "", "PDF content not available"
    
    combined_content = ""
    section_titles = []
    
    for section_key in section_keys:
        if section_key in pdf_content["sections"]:
            section_data = pdf_content["sections"][section_key]
            combined_content += f"\n\n=== {section_data['title']} (Pages {section_data['page_range']}) ===\n"
            combined_content += section_data["content"]
            section_titles.append(section_data['title'])
    
    return combined_content, ", ".join(section_titles)


add_fullscreen_background()

# Load PDF content
pdf_content = load_pdf_content()

# Main title
st.title("Bishop's University Academic Calendar Assistant")

# Only show errors if PDF fails to load
if pdf_content["status"] == "file_not_found":
    st.markdown(
        f"""
        <div class="pdf-error">
            ❌ {pdf_content["error"]}
        </div>
        """,
        unsafe_allow_html=True
    )
elif pdf_content["status"] == "error":
    st.markdown(
        f"""
        <div class="pdf-error">
            ⚠️ {pdf_content["error"]}
        </div>
        """,
        unsafe_allow_html=True
    )

# Multi-Model mode enabled by default
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API key not found. Please configure it in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=openai_key)


def summarize_with_model(text, model):
    """Summarize text using specified model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following academic content from Bishop's University calendar. Focus on key information that would be most useful for students."},
                {"role": "user", "content": text[:4000]}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"


def compute_similarity_matrix(summaries):
    """Compute similarity matrix between summaries"""
    if len(summaries) < 2:
        return np.array([[1.0]])
    vect = TfidfVectorizer().fit_transform(summaries)
    return cosine_similarity(vect)


def find_central_summary(matrix, summaries):
    """Find the most central summary based on similarity scores"""
    scores = matrix.sum(axis=1)
    return summaries[int(np.argmax(scores))]


def answer_question(question, relevant_content, section_titles):
    """Answer question based on relevant PDF content"""
    prompt = f"""You are a helpful academic advisor at Bishop's University.
Use the following content from the Bishop's University Academic Calendar 2025-2026 to answer the student's question.
The content comes from these sections: {section_titles}

Academic Calendar Content:
{relevant_content[:5000]}

Student Question: {question}

Please provide a helpful and accurate answer based on the calendar information. If the answer involves specific page numbers, deadlines, or requirements, please include those details:"""
    
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
    # Find relevant sections for the subject
    relevant_sections = identify_relevant_sections(subject, top_k=2)
    relevant_content, section_titles = get_section_content(relevant_sections)
    
    if not relevant_content.strip():
        return f"Unable to find relevant information about '{subject}' in the academic calendar."
    
    prompt = f"""You are a helpful academic advisor at Bishop's University. 
Please provide a comprehensive summary about "{subject}" based on the following content from the Bishop's University Academic Calendar 2025-2026.
The content comes from these sections: {section_titles}

Calendar Content:
{relevant_content[:4000]}

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
        if pdf_content["status"] == "success":
            # Find relevant sections
            relevant_sections = identify_relevant_sections(question, top_k=2)
            relevant_content, section_titles = get_section_content(relevant_sections)
            
            # Display matched sections
            st.markdown(
                f"""
                <div class="section-indicator">
                    📍 Matched Sections: {section_titles}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if relevant_content.strip():
                # Multi-model approach by default
                models = ["gpt-3.5-turbo", "gpt-4"]
                summaries = [summarize_with_model(relevant_content, m) for m in models]
                matrix = compute_similarity_matrix(summaries)
                central = find_central_summary(matrix, summaries)

                answer = answer_question(question, central, section_titles)
                st.markdown(
                    f"""
                    <div class="summary-section">
                        <p>{answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="summary-section">
                        <p>Unable to find relevant information for your question. Please try rephrasing your question or check if it relates to content covered in the academic calendar.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="summary-section">
                    <p>PDF file is not available. Please ensure 'BU-AcCal-2025-2026-REV-July-2.pdf' is in the same directory as this application.</p>
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
        if pdf_content["status"] == "success":
            with st.spinner("Generating summary..."):
                # Find relevant sections
                relevant_sections = identify_relevant_sections(subject_input, top_k=2)
                _, section_titles = get_section_content(relevant_sections)
                
                # Display matched sections
                st.markdown(
                    f"""
                    <div class="section-indicator">
                        📍 Relevant Sections: {section_titles}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Multi-model approach by default
                models = ["gpt-3.5-turbo", "gpt-4"]
                summaries = [summarize_calendar_content(subject_input, model) for model in models]
                matrix = compute_similarity_matrix(summaries)
                central_summary = find_central_summary(matrix, summaries)
                
                st.markdown(
                    f"""
                    <div class="summary-section">
                        <p>{central_summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="summary-section">
                    <p>PDF file is not available. Please ensure 'BU-AcCal-2025-2026-REV-July-2.pdf' is in the same directory as this application.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer
st.markdown(
    """
    <div class="footer">
        Copyright © BU. Version 0.3 - Smart Section-Based Processing. Last update August 2025
    </div>
    """,
    unsafe_allow_html=True
)
