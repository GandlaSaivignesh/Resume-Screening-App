This project is a Streamlit-based Resume Screening System that helps recruiters automatically compare resumes against job descriptions and rank candidates based on their relevance.

1. Imports and Libraries

The app uses a mix of NLP, visualization, and web-app libraries:

NLP/ML: sklearn (TF-IDF, cosine similarity), regex for keyword extraction.

Visualization: matplotlib, plotly, wordcloud.

UI/Frontend: streamlit for the dashboard.

File Handling: pandas, PIL, os, time.

2. Similarity Calculation

The function calculate_similarity(resume_text, job_description) generates a composite score using multiple factors:

TF-IDF Similarity (40%)
Measures semantic similarity between resume text and job description.

Keyword Matching (30%)
Extracts important keywords (excluding stopwords) and checks overlap.

Skills Matching (20%)
Uses regex patterns to identify technical & business skills.
Example: Python, Java, SQL, AWS, Agile, Data Analysis, etc.

Length Appropriateness (10%)
Ensures the resume isn’t too short or disproportionately long compared to the job description.

The final weighted score is scaled to 0–100%.

3. Keyword & Skill Extraction

extract_keywords(): Captures important non-stopword terms from text.

extract_skills(): Matches text against predefined regex patterns for technical/business skills.

This allows the system to highlight what specific skills and keywords are common between resumes and job descriptions.

4. User Inputs

The app supports two input methods for both resumes and job descriptions:

File Upload (.pdf, .docx, .txt)

Manual Input (text area with --- as separator)

Uploaded/entered content is stored in structured lists, then converted into Pandas DataFrames for processing.

5. Job Description Selection

If multiple job descriptions are uploaded:

The user selects one via a dropdown (st.selectbox).

Optionally, they can view the full text of the job description.

6. Score Calculation & Ranking

Each resume is compared against the selected job description.

Scores are calculated with caching (@st.cache_data) for efficiency.

Results are sorted, ranked, and stored in Ranked_resumes.

7. Results Display

The results are shown in multiple visual formats:

Top Ranked Resumes (Table)
A Plotly table displaying Rank, Resume Name, and Score %.

Score Distribution (Bar Chart)
A color-coded bar chart showing how each resume scored.

Detailed Results (Expander View)
Expandable sections for each resume showing:

Resume Content (truncated if long)

Job Description (truncated if long)

8. Download Results

A summary of ranks, names, and scores is available for download as CSV.

File is named with the selected job description for context.

9. Summary Statistics

At the end, the app displays quick insights:

Total resumes screened.

Average similarity score.

Highest score among all resumes.
