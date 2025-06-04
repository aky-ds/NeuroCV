import os
from fuzzywuzzy import fuzz
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import spacy
import re
from langchain_groq import ChatGroq
import groq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Compute similarity score (scaled to 5)
def compute_similarity(resume_text, job_desc):
    embeddings1 = model.encode(resume_text, convert_to_tensor=True)
    embeddings2 = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return round(similarity * 5, 2)

# Extract education
def extract_education(text):
    degrees = ["phd", "master", "bachelor", "bs", "ms", "mba", "bsc", "msc", "m.tech", "b.tech"]
    text_lower = text.lower()
    for degree in degrees:
        if degree in text_lower:
            return degree.upper()
    return "Not Found"

# Extract years of experience
def estimate_experience(text):
    # Create a proper prompt template
    prompt_template = PromptTemplate.from_template("""
    First of all check the skills and note it and if the experience is according to the skills then Perform the following task else return 0:
    From the resume below, estimate the total years of professional work experience.
    Consider all job roles and their start and end dates.
    Return only a single number (e.g., 4.5 or 6) if it is not resume or you find no answer then return as 0 only give me the number as 0.5 or anything else not explanation:

    Resume:
    {text}
    """)
    
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0
    )
    
    # Create a proper chain
    chain = prompt_template | llm | StrOutputParser()
    
    # Invoke the chain with the input
    result = chain.invoke({"text": text})
    return result

# Match certifications from a provided list
def match_certifications(text, cert_list, threshold=60):
    found = []
    resume_text = re.sub(r'\s+', ' ', text.lower()).strip()

    for cert in cert_list:
        cert_cleaned = cert.lower().strip()
        if cert_cleaned in resume_text:
            found.append(cert)
    return found


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files["resume"]
        job_desc = request.form["job_desc"]
        certs = request.form.get("certifications", "")
        certs = certs.split(',') if certs else []

        try:
            resume_text = extract_text_from_pdf(resume_file)

            similarity_score = compute_similarity(resume_text, job_desc)
            education = extract_education(resume_text)
            experience = float(estimate_experience(resume_text))
            found_certs = match_certifications(resume_text, certs)

            # Score components
            education_score = 1 if education != "Not Found" else 0
            experience_score = 2 if experience >= 2 else 0
            certification_score = 3 if len(found_certs) >= 1 else 0
            similarity_scaled = min(similarity_score, 5)  # max 5

            # Weighted total score (out of 10)
            total_score = round(similarity_scaled * (4 / 5) + certification_score + experience_score + education_score, 2)

            # Final decision
            result = "✅ Selected" if total_score >= 6 else "❌ Not Selected"

            return render_template("result.html",
                                   score=similarity_score,
                                   education=education,
                                   experience=experience,
                                   certs=found_certs,
                                   result=result,
                                   total_score=total_score)
        except Exception as e:
            return f"Error occurred: {e}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
