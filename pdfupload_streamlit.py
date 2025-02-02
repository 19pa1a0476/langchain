
import os
import tempfile
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_upstage import UpstageLayoutAnalysisLoader
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


#UpStage Solar API Key
os.environ["UPSTAGE_API_KEY"] = "UP_STAGE_API_KEY"

# Azure Open-AI Key
os.environ["AZURE_OPENAI_API_KEY"] = "AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "AZURE_OPENAI_API_VERSION"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
embeddings = AzureOpenAIEmbeddings(model = 'Model')

#Qdrant Vector Store API-KEY & Endpoint
qdrant_url = "Qdrant_Vector_Store_URL"
qdrant_api_key = "Qdrant_API_KEY"


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please provide the answer to the user's questions based on following content: \n {found_docs}",
        ),
        ("human", "{query}"),
    ]
)


#Function to get the response from Open-AI
def open_ai_handler(query, found_docs):

    llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],)

    parser = StrOutputParser()
    chain = prompt | llm | parser

    output = chain.invoke(
        {
            "found_docs": found_docs,
            "query": query,
        }
    )

    return output


# Function to format response
def format_response(response):
    # Define color coding
    color_coding = {
        'digit': 'color: green; font-weight: bold;',
        'alpha': 'color: white; font-weight: bold;',
        'special': 'color: blue; font-weight: bold;',
        'default': 'color: white;'
    }

    formatted_text = ""
    special_characters = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    for char in response:
        if char.isdigit():
            formatted_text += f'<span style="{color_coding["digit"]}">{char}</span>'
        elif char.isalpha():
            formatted_text += f'<span style="{color_coding["alpha"]}">{char}</span>'
        elif char in special_characters:
            formatted_text += f'<span style="{color_coding["special"]}">{char}</span>'
        else:
            formatted_text += f'<span style="{color_coding["default"]}">{char}</span>'

    return formatted_text


# Function to load the pdf and store in the vector store
def process_uploaded_pdf(file):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_location = tmp_file.name
        
        # Use PyPdfLoader to load and process the PDF
        loader = PyPDFLoader(tmp_location)

        #Use this for accauracy more and cos support
        # loader = UpstageLayoutAnalysisLoader(tmp_location, use_ocr = True, output_type="text")    
        
        pages = loader.load_and_split()

        # Perform operations on 'pages' here
        qdrant = Qdrant.from_documents(
            pages,
            embeddings,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name="Collection Name",)
        
        st.write(f"Document processed successfully!")

    except Exception as e:
        st.write(f"Error occurred: {e}")

    finally:
        # Clean up: Delete the temporary file after use
        if tmp_location and os.path.exists(tmp_location):
            os.remove(tmp_location)
    

# Title of the app
st.title("SAP Test Case Generator Text2Code Bot")

# Initialize session state for pdf_text if not already initialized
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file")

if uploaded_file is not None:
    # Read the PDF and store text in session state
    if st.session_state.pdf_text is None:
        with st.spinner("Processing the File..."):
            st.write(f"File uploaded: {uploaded_file.name}")
            # Process the uploaded PDF file
            process_uploaded_pdf(uploaded_file)
            st.session_state.pdf_text ="SUCCESS"

# Check if PDF text is available in session state
if st.session_state.pdf_text is not None:
    st.empty()
    # Display form for user query
    with st.form(key='query_form'):
        query = st.text_area("Enter your query:", height=200)
        submit_button = st.form_submit_button(label='Submit Query')

    if submit_button:
        st.session_state.response = ""
        response_placeholder = st.empty()
        with st.spinner("Generating Response..."):
            qdrant = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,)

            query_embedding = embeddings.embed_query(query)

            found_docs = qdrant.search(
                collection_name="BRD_Document",
                query_vector="x1x",   # make x1x as non string
                limit=3,)
            
            response = open_ai_handler(query, found_docs)
            st.session_state.response = "SUCCESS"
        
            # Display response if available
            if st.session_state.response:
                formatted_response = f"OpenAI Response:\n\n{response}"
                response_placeholder.text(formatted_response)
                
else:
    st.write("Please upload a PDF file.")

