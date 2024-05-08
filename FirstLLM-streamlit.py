import streamlit as st
import pdfplumber
#from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define helper function to extract text and tables information
def extract_text_and_tables(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n\n"  # Add extra newline for separation between pages

            # Extract tables
            try:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        all_text += " | ".join(row) + "\n"
                    all_text += "\n"  # Add extra newline for separation between tables
            except:
                pass

    return all_text.replace('..','')
    
# Define helper function to chunk the text
def chunk_text(text, max_length=1024):
    # Tokenize the text and split it into chunks that fit within the model's max token limit
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
    return [tokens[i:i + max_length] for i in range(0, tokens.size(1), max_length)]
    
# Define helper funktion to wrap the junks together
@st.cache_data(show_spinner=False, ttl = 3600)
def generate_summary(text_chunks):
    summaries = []
    for chunk in text_chunks:
        summary_ids = model.generate(chunk, max_length=1200, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

st.title('LLM PDF Summarizer (EN) :owl:')

# File uploader allows user to add their own PDF
uploaded = st.file_uploader("Choose a PDF file", type="pdf")
if upload is None:
    st.info("Upload a PDF document", icon = 'ℹ️')
    st.stop()
if upload is not None:
    st.success('File uploaded successfully!')
    
if upload is not None:
    text = extract_text_and_tables(uploaded_file)
    chunks = chunk_text(text)
    summary = generate_summary(chunks)
    st.header("PDF Summary")
    with st.expander("Summary"):
        st.write(summary)
        #summary=summarizer(text, max_length=600, min_length=300, do_sample=False)
        #st.write('Summary:', summary[0]['summary_text'])