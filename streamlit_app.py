import validators, streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Streamlit app
st.subheader('Summarize URL')

# Get OpenAI API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Get model selection and URL
with st.sidebar:
    model = st.selectbox("OpenAI chat model", ("llama-3.1-8b-instant", "llama-3.3-70b-versatile"))
    st.caption("*For longer articles, choose llama-3.3-70b-versatile.*")

url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not url.strip():
        st.error("Please provide a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url])
                data = loader.load()
                
                # Initialize the ChatGroq module and create a chain
                llm = ChatGroq(temperature=0, model=model, groq_api_key=groq_api_key)
                prompt_template = """Write a summary of the following in 250-300 words:
                    
                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = prompt | llm | StrOutputParser()
                
                # Combine all document content
                article_text = "\n\n".join(doc.page_content for doc in data)
                summary = chain.invoke({"text": article_text})

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")