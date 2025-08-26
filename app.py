import matplotlib.colors as mcolors
import numpy as np
from wordcloud import WordCloud
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced similarity calculation with multiple factors
def calculate_similarity(resume_text, job_description):
    """Calculate comprehensive similarity score between resume and job description"""
    if not resume_text or not job_description:
        return 0.0
    
    try:
        # Base TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([resume_text.lower(), job_description.lower()])
        base_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Keyword matching score
        job_keywords = extract_keywords(job_description)
        resume_keywords = extract_keywords(resume_text)
        
        # Calculate keyword overlap
        keyword_overlap = len(set(job_keywords) & set(resume_keywords))
        keyword_score = (keyword_overlap / len(job_keywords)) if job_keywords else 0
        
        # Length normalization
        resume_length = len(resume_text.split())
        job_length = len(job_description.split())
        length_ratio = min(resume_length / max(job_length, 1), 1.5)  # Allow up to 1.5x job length
        
        # Skills matching (basic)
        job_skills = extract_skills(job_description)
        resume_skills = extract_skills(resume_text)
        skills_overlap = len(set(job_skills) & set(resume_skills))
        skills_score = (skills_overlap / len(job_skills)) if job_skills else 0
        
        # Weighted final score
        final_score = (
            base_similarity * 0.4 +           # 40% TF-IDF
            keyword_score * 0.3 +             # 30% keyword matching
            skills_score * 0.2 +              # 20% skills matching
            min(length_ratio, 1.0) * 0.1      # 10% length appropriateness
        )
        
        return round(final_score * 100, 2)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def extract_keywords(text):
    """Extract important keywords from text"""
    # Common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Return top keywords
    return list(set(keywords))[:20]

def extract_skills(text):
    """Extract skills from text"""
    # Common technical skills and keywords
    skill_patterns = [
        r'\b(?:python|java|javascript|sql|html|css|react|angular|vue|node|django|flask|spring|aws|azure|gcp|docker|kubernetes|git|agile|scrum|machine learning|data science|ai|ml|nlp|tableau|power bi|excel|powerpoint|word|outlook|crm|sap|oracle|mysql|postgresql|mongodb|redis|elasticsearch|jenkins|ci/cd|devops|linux|windows|macos|ios|android|swift|kotlin|go|ruby|php|c#|c\+\+|\.net|typescript|scala|spark|hadoop|tensorflow|pytorch|keras|pandas|numpy|scikit-learn|matplotlib|seaborn|plotly|d3\.js|react\.js|vue\.js|angular\.js)\b',
        r'\b(?:project management|business analysis|data analysis|business intelligence|quality assurance|software development|web development|mobile development|frontend|backend|full stack|ui/ux|design|testing|debugging|troubleshooting|optimization|performance|security|scalability|architecture|database|api|rest|microservices|cloud|saas|paas|iaas|automation|scripting|integration|deployment|monitoring|logging|analytics|reporting|dashboard|visualization)\b'
    ]
    
    skills = []
    text_lower = text.lower()
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        skills.extend(matches)
    
    return list(set(skills))

# Logo handling with error checking
try:
    image = Image.open('Images/logo.png')
    st.image(image, use_column_width=True)
except:
    st.warning("Logo image not found at Images/logo.png")

st.title("Resume Screening with Upload")

# File upload section
st.header("Upload Resumes and Job Descriptions")
st.markdown("---")

# Resume upload section
st.subheader("📄 Upload Resumes")
uploaded_resumes = st.file_uploader(
    "Choose resume files", 
    type=['pdf', 'docx', 'txt'], 
    accept_multiple_files=True,
    help="Upload multiple resume files (PDF, DOCX, TXT)"
)

# Job description upload section
st.subheader("📋 Upload Job Descriptions")
uploaded_jobs = st.file_uploader(
    "Choose job description files", 
    type=['pdf', 'docx', 'txt'], 
    accept_multiple_files=True,
    help="Upload multiple job description files (PDF, DOCX, TXT)"
)

# Manual input section
st.subheader("✏️ Or Enter Manually")
col1, col2 = st.columns(2)

with col1:
    manual_resumes = st.text_area(
        "Enter Resumes (one per line):",
        height=200,
        placeholder="Paste resume content here...\nUse --- to separate multiple resumes"
    )

with col2:
    manual_jobs = st.text_area(
        "Enter Job Descriptions (one per line):",
        height=200,
        placeholder="Paste job descriptions here...\nUse --- to separate multiple descriptions"
    )

# Process uploaded files
def process_uploaded_files(uploaded_files):
    """Process uploaded files and return text content"""
    contents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            # For PDF files, you would need PyPDF2 or similar
            contents.append(f"PDF content from {file.name}")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # For DOCX files, you would need python-docx
            contents.append(f"DOCX content from {file.name}")
        else:
            # For text files
            contents.append(file.read().decode('utf-8'))
    return contents

def process_manual_input(text_input, separator="---"):
    """Process manual text input"""
    if not text_input.strip():
        return []
    return [item.strip() for item in text_input.split(separator) if item.strip()]

# Process all inputs
resumes_data = []
jobs_data = []

if uploaded_resumes:
    resume_contents = process_uploaded_files(uploaded_resumes)
    resumes_data.extend([{"Name": f"Resume_{i+1}", "Content": content, "TF_Based": content} 
                        for i, content in enumerate(resume_contents)])

if manual_resumes:
    manual_resume_list = process_manual_input(manual_resumes)
    resumes_data.extend([{"Name": f"Manual_Resume_{i+1}", "Content": content, "TF_Based": content} 
                        for i, content in enumerate(manual_resume_list)])

if uploaded_jobs:
    job_contents = process_uploaded_files(uploaded_jobs)
    jobs_data.extend([{"Name": f"Job_{i+1}", "Context": content, "TF_Based": content} 
                     for i, content in enumerate(job_contents)])

if manual_jobs:
    manual_job_list = process_manual_input(manual_jobs)
    jobs_data.extend([{"Name": f"Manual_Job_{i+1}", "Context": content, "TF_Based": content} 
                     for i, content in enumerate(manual_job_list)])

# Only proceed if we have data
if not resumes_data or not jobs_data:
    st.warning("Please upload or enter both resumes and job descriptions to proceed.")
    st.stop()

# Convert to DataFrames
Resumes = pd.DataFrame(resumes_data)
Jobs = pd.DataFrame(jobs_data)

st.success(f"Loaded {len(resumes_data)} resumes and {len(jobs_data)} job descriptions!")

############# JOB DESCRIPTION SELECTION ##############
st.markdown("---")
st.header("🎯 Select Job Description for Screening")

if len(Jobs['Name']) > 1:
    job_option = st.selectbox(
        "Choose a job description to screen resumes against:",
        options=Jobs['Name'],
        help="Select which job description to use for screening"
    )
    selected_job_index = Jobs[Jobs['Name'] == job_option].index[0]
else:
    selected_job_index = 0
    st.info(f"Using job description: {Jobs['Name'][0]}")

# Display selected job description
if st.checkbox("Show selected job description"):
    st.subheader("Selected Job Description:")
    st.write(Jobs['Context'][selected_job_index])

############### SCORE CALCULATION ####################
st.markdown("---")
st.header("📊 Screening Results")

@st.cache_data
def calculate_scores(resumes, job_description, idx):
    scores = []
    for x in range(resumes.shape[0]):
        score = calculate_similarity(
            str(resumes['TF_Based'][x]), 
            str(job_description['TF_Based'][idx])
        )
        scores.append(score)
    return scores

Resumes['Scores'] = calculate_scores(Resumes, Jobs, selected_job_index)
Ranked_resumes = Resumes.sort_values(by=['Scores'], ascending=False).reset_index(drop=True)
Ranked_resumes['Rank'] = range(1, len(Ranked_resumes) + 1)

#################### RESULTS DISPLAY ##################
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top Ranked Resumes")
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=["Rank", "Name", "Score (%)"],
                    fill_color='#00416d',
                    font=dict(color='white', size=14)),
        cells=dict(values=[Ranked_resumes.Rank, Ranked_resumes.Name, Ranked_resumes.Scores],
                   fill_color='#d6e0f0')
    )])
    fig1.update_layout(width=400, height=400)
    st.write(fig1)

with col2:
    st.subheader("📊 Score Distribution")
    fig2 = px.bar(Ranked_resumes,
                  x='Name', y='Scores',
                  color='Scores',
                  color_continuous_scale='viridis',
                  title="Resume Scores")
    fig2.update_layout(height=400)
    st.write(fig2)

#################### DETAILED RESULTS ##################
st.markdown("---")
st.header("📋 Detailed Results")

# Show detailed comparison
for idx, row in Ranked_resumes.iterrows():
    with st.expander(f"#{row['Rank']} - {row['Name']} (Score: {row['Scores']}%)"):
        st.write("**Resume Content:**")
        st.write(row['Content'][:500] + "..." if len(row['Content']) > 500 else row['Content'])
        
        st.write("**Job Description:**")
        st.write(Jobs['Context'][selected_job_index][:500] + "..." if len(Jobs['Context'][selected_job_index]) > 500 else Jobs['Context'][selected_job_index])

#################### DOWNLOAD RESULTS ##################
st.markdown("---")
st.header("💾 Download Results")

# Prepare download data
results_df = Ranked_resumes[['Rank', 'Name', 'Scores']].copy()
results_df['Job_Description'] = Jobs['Name'][selected_job_index]

csv = results_df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name=f"resume_screening_results_{Jobs['Name'][selected_job_index].replace(' ', '_')}.csv",
    mime="text/csv"
)

# Summary statistics
st.markdown("---")
st.header("📊 Summary Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Resumes", len(Ranked_resumes))
with col2:
    st.metric("Average Score", f"{Ranked_resumes['Scores'].mean():.2f}%")
with col3:
    st.metric("Highest Score", f"{Ranked_resumes['Scores'].max():.2f}%")
