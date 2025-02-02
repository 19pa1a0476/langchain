import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_upstage import UpstageLayoutAnalysisLoader
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.document_loaders import UnstructuredPDFLoader
import requests 
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader


#UpStage Solar API Key
os.environ["UPSTAGE_API_KEY"] = "UP_STAGE_API_KEY"

#Qdrant Vector Store Api Keys
qdrant_url = "Qdrant_Vector_Store_URL"
qdrant_api_key = "Qdrant_API_KEY"

#Setting up AzureOpenAPI Key - Please replace your open api key
os.environ["AZURE_OPENAI_API_KEY"] = "AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "AZURE_OPENAI_API_VERSION"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
embeddings = AzureOpenAIEmbeddings(model = 'Model')


system_prompt = (
    "Instructions:Do not explain.Provide only executable code. Requirement: Generate code for the usedr input in domain-specific language (DSL) used by IBM Cognos, specifically within Cognos Report Studio or Cognos Analytics for creating custom calculations and data expressions.To fetch the value of a particular column Measure keyword is used. Syntax example: Measure.[Stat Raw History]= if (~isnull(Measure.[Actual])) then Measure.[Actual];"
    "{found_docs}"
)
# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "{query}"),
    ]
)
def pdfUploader(file_path):
    # loader = UnstructuredPDFLoader(file_path)
    
    # Use PyPdfLoader to load and process the PDF
    loader = PyPDFLoader(file_path)

    #Use this for accauracy more and cos support
    # loader = UpstageLayoutAnalysisLoader(tmp_location, use_ocr = True, output_type="text"
    pages = loader.load_and_split()
    # Perform operations on 'pages' here
    qdrant = Qdrant.from_documents(
        pages,
        embeddings,
        url=qdrant_url,
        prefer_grpc=True,
        api_key=qdrant_api_key,
        collection_name = "Collection Name",)

# Function to load the pdf and store in the vector store
def process_uploaded_pdf():
    try:
        file_path3 = "File Path"
        pdf_paths = [file_path3]
        # Use UpStageLayoutLoader to load and process the PDF
        for pdf_path in pdf_paths:
            pdfUploader(pdf_path)
        print(f"Document processed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")

def askQuery(query,pdf_collection):
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,)

    embeddings = AzureOpenAIEmbeddings(model = 'Model')

    query_embedding = embeddings.embed_query(query)

    found_docs = qdrant.search(
        collection_name=pdf_collection,
        query_vector=query_embedding,
        limit=3,)

    print(found_docs)

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
    
syle = """
    <style>
 
    .stButton>button { background-color: #FF6F61; color: white; border-radius: 5px; }
    .stTextArea>textarea { border-radius: 5px; }
 
    section>[data-testid="stSidebarContent"] { background: #FDC70D; background-color: #FDC70D; }
 
    section>[data-testid="stMarkdownContainer"] { font-size: 10px; }
 
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem; font-weight: bold; }
 
    .stTabs [data-baseweb="tab-list"] { font-size: 1rem; }
 
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem; font-weight: bold; }
 
    .stMarkdown [data-testid="stMarkdownContainer"] p { font-size: 1rem; font-weight: bold; color:rgb(255, 75, 75);}
 
    </style>
    """
st.markdown(syle, unsafe_allow_html=True)
sideb = st.sidebar
sideb.title("Domain-Specific Language (DSL) Text2Code Bot")
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

campaign_details = sideb.text_area("Enter the Requirements", height=160)
campaign_collection = sideb.text_area("Enter the Collection", height=30)
if campaign_collection=="":
    campaign_collection = "Collection Name"
sidebarbutton = sideb.button("Generate Code")
sidebarbutton_test = sideb.button("Generate Test Data")
 
if sidebarbutton:
    with st.container(border=None):
        st.markdown("Generated Code")
        if campaign_details != "" and st.session_state.pdf_text is None:
           process_uploaded_pdf()
           st.session_state.pdf_text = "SUCCESS"
        if campaign_details != "" and st.session_state.pdf_text is "SUCCESS":
            results = askQuery(campaign_details,campaign_collection)
            st.code(results)

