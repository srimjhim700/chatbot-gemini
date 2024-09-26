import streamlit as st
from neo4j import GraphDatabase
import io
from fpdf import FPDF
import base64
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
import re

NEO4J_URI='neo4j+s://a53f2d90.databases.neo4j.io'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='CypK_rg3X5tWnr8R7FCwxD8Rpv5fQzFDTwqtz7haR0E'
GOOGLE_API_KEY="AIzaSyAu6CmYGRGSVuRpP8LkT37iyhNL_rqt2kI"
#GROQ_API_KEY = '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'

GROQ_API_KEY='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'


llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Gemma2-9b-It")


def clean_text(text):
  """Cleans the given text by removing noise like special characters and extra whitespace.

  Args:
    text: The text to be cleaned.

  Returns:
    The cleaned text.
  """

  # Remove special characters
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

  # Remove extra whitespace
  text = re.sub(r"\s+", " ", text)

  return text

llm_transformer=LLMGraphTransformer(llm=llm)





# Initialize the Google Gemini LLM (Gemini 1.5 Pro)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,  # Adjust for response variation
    max_tokens=1000,  # Set maximum token limit for response length
    timeout=None,
    max_retries=2,
    api_key=GOOGLE_API_KEY,
)

# Function to generate the legal response using LLM and Neo4j results
def generate_legal_response_llm(user_input, cleaned_data):
    # Format the cleaned Neo4j data into readable case details for the prompt
    formatted_data = ""
    for item in cleaned_data:
        node_id = item['node']['id'] if item['node'] and item['node']['id'] else 'None'
        node_label = item['node']['label'] if item['node'] and item['node']['label'] else 'None'
        related_node_id = item['related_node']['id'] if item['related_node'] and item['related_node']['id'] else 'None'
        related_node_label = item['related_node']['label'] if item['related_node'] and item['related_node']['label'] else 'None'
        relationship = item['relationship'] if item['relationship'] else 'None'

        # Only include non-'None' relationships
        if node_id != 'None' and related_node_id != 'None' and relationship!='None':
            formatted_data += f"Node: {node_id} ({node_label}) is related to {related_node_id} ({related_node_label}) by {relationship}.\n"

    # Define the LLM system prompt and user input
    messages = [
        (
            "system",
            """
            You are an AI legal assistant tasked with analyzing the following case based on Indian judiciary principles. 
            The case details provided come from a combination of official legal documents (FIR, Charge Sheet, Crime History) and a pre-built knowledge graph, containing relevant data from the Bhartiya Nyay Sanhita (Indian Penal Code), as well as other legal references like the Criminal Procedure Code (CrPC) and past legal precedents.

            Use this information and consider the following guidelines while generating your answer:

            - Ensure the response adheres to the current legal framework in India, including the Bhartiya Nyay Sanhita, the Criminal Procedure Code (CrPC), and any relevant judicial precedents.
            - Refer to the retrieved data from the Neo4j knowledge graph, including specific nodes like FIR details, Charge Sheet details, and Crime History.
            - Focus on providing an explanation that is legally sound and justified based on the given data.
            """,
        ),
        (
            "human",
            f"""
            Here are the case details for your reference:
            
            {user_input}

            Additional Data from Neo4j:
            {formatted_data}

            Based on these details, please generate:
            1. A concise summary of the case.
            2. A bail eligibility score based on the Bhartiya Nyay Sanhita and CrPC.
            3. Court eligibility based on jurisdiction and crime severity.
            4. Recommended bail type (Regular, Anticipatory, or Interim) with reasoning.
            5. A detailed legal description, including relevant sections from Indian law (e.g., IPC, CrPC), and how they apply to this case.
            """
        ),
    ]
    print(messages)

    # Invoke the LLM with the messages
    ai_msg = llm.invoke(messages)

    # Return the generated legal response
    return ai_msg




# Call the function to get the legal response



# Function to query the Neo4j database and retrieve relevant information
def query_neo4j_graph(graph_documents):
    # Assuming we have an active Neo4j session
    uri = NEO4J_URI
    username = "neo4j"
    password = NEO4J_PASSWORD
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Store results in a list
    retrieved_data = []
    
    # Open a Neo4j session
    with driver.session() as session:
        for doc in graph_documents:
            # Extract node IDs, types, and relationships from the document
            for node in doc.nodes:
                node_id = node.id
                node_type = node.type
                
                # Define Cypher query to match nodes and relationships based on the document structure
                query = f"""
                MATCH (n:{node_type} )
                OPTIONAL MATCH (n)-[r]-(related)
                RETURN n, r, related
                """
                
                # Execute the query
                result = session.run(query)
                
                # Process each record in the result set
                for record in result:
                    # Get node and relationships from the record
                    node_data = record.get("n")
                    relationship_data = record.get("r")
                    related_node = record.get("related")
                    
                    # Add the data to the list in a structured format
                    retrieved_data.append({
                        "node": node_data,
                        "relationship": relationship_data,
                        "related_node": related_node
                    })
    

    
    # Close the Neo4j driver connection
    driver.close()
    
    return retrieved_data

# Example usage: Assume graph_documents is a list of dictionaries with nodes and relationships


def parse_retrieved_data(retrieved_data):
    parsed_data = []

    # Iterate over each record in the retrieved_data list
    for record in retrieved_data:
        node = record.get('node')
        relationship = record.get('relationship')
        related_node = record.get('related_node')

        # Extract necessary information from the node (if node is not None)
        if node:
            node_info = {
                'id': node['id'],
                'label': list(node.labels)[0] if node.labels else None  # Extract label if available
            }
        else:
            node_info = None

        # Extract relationship type (if relationship is not None)
        relationship_info = relationship.type() if relationship else None

        # Extract necessary information from the related node (if related_node is not None)
        if related_node:
            related_node_info = {
                'id': related_node['id'],
                'label': list(related_node.labels)[0] if related_node.labels else None  # Extract label if available
            }
        else:
            related_node_info = None

        # Append to the parsed data
        parsed_data.append({
            'node': node_info,
            'relationship': relationship_info,
            'related_node': related_node_info
        })

    return parsed_data

# Example usage with the retrieved_data variable


def generate_pdf(response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Case Analysis", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(200, 10, txt=response)
    return pdf.output(dest='S').encode('latin-1')
# Function to generate PDF and return binary content
def generate_pdf_report(summary, bail_score, court_eligibility, bail_type, law_description):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Case Report", ln=True, align="C")

    pdf.ln(10)
    pdf.cell(200, 10,txt="Dsiclaimer: This report is generated using AI")
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Case Summary:", ln=True)
    pdf.multi_cell(200, 10, txt=summary)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Bail Eligibility Score:", ln=True)
    pdf.multi_cell(200, 10, txt=bail_score)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Court Eligibility:", ln=True)
    pdf.multi_cell(200, 10, txt=court_eligibility)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Bail Type:", ln=True)
    pdf.multi_cell(200, 10, txt=bail_type)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Description of the Law:", ln=True)
    pdf.multi_cell(200, 10, txt=law_description)

    pdf_output = "case_report.pdf"
    
    return pdf.output(dest='S').encode('latin-1')



# Function to convert PDF binary content to base64
def pdf_to_base64(pdf_content):
    return base64.b64encode(pdf_content).decode('utf-8')


def main():
    st.set_page_config("Legal Case Analyzer")
    st.header("Legal Case Analyzer with Neo4j💼")

    st.write("Enter case details to get relevant laws and a detailed report.")

    with st.sidebar:
        st.title("Input:")
        cnr_number = st.text_input("Enter CNR Number:")
        fir_details = st.text_area("Enter FIR Details:")
        charge_sheet_details = st.text_area("Enter Charge Sheet Details:")
        witness_statements = st.text_area("Enter Witness Statements:")
        medical_info = st.text_area("Enter Medical Reasons (if any):")
        crime_history = st.text_area("Enter Crime History:")

        user_input = f"""
        CNR: CNR-87654321
        FIR: {fir_details.strip()}
        Charge Sheet: {charge_sheet_details.strip()}
        Crime History: {crime_history.strip()}
        """
        cleaned_ip = clean_text(user_input)
        documents = [Document(page_content=cleaned_ip) ]
        graph_documents=llm_transformer.convert_to_graph_documents(documents)
        retrieved_data = query_neo4j_graph(graph_documents)
        cleaned_data = parse_retrieved_data(retrieved_data)
        legal_response = generate_legal_response_llm(user_input, cleaned_data)
        legal_response=legal_response.content

    if st.button("Analyze Case"):
        with st.spinner("Processing the case..."):
            summary_pattern = r"\*\*1\. Case Summary:\*\*(.*?)\*\*2\. Bail Eligibility Score:\*\*"
            bail_score_pattern = r"\*\*2\. Bail Eligibility Score:\*\*(.*?)\*\*3\. Court Eligibility:\*\*"
            court_eligibility_pattern = r"\*\*3\. Court Eligibility:\*\*(.*?)\*\*4\. Recommended Bail Type:\*\*"
            bail_type_pattern = r"\*\*4\. Recommended Bail Type:\*\*(.*?)\*\*5\. Detailed Legal Description:\*\*"
            law_description_pattern = r"\*\*5\. Detailed Legal Description:\*\*(.*?)\*\*Disclaimer:\*\*"

            # Extract using the patterns
            summary_match = re.search(summary_pattern, legal_response, re.DOTALL)
            bail_score_match = re.search(bail_score_pattern, legal_response, re.DOTALL)
            court_eligibility_match = re.search(court_eligibility_pattern, legal_response, re.DOTALL)
            bail_type_match = re.search(bail_type_pattern, legal_response, re.DOTALL)
            law_description_match = re.search(law_description_pattern, legal_response, re.DOTALL)

            # Assign the matched groups to variables
            summary = summary_match.group(1).strip() if summary_match else "N/A"
            bail_score = bail_score_match.group(1).strip() if bail_score_match else "N/A"
            court_eligibility = court_eligibility_match.group(1).strip() if court_eligibility_match else "N/A"
            bail_type = bail_type_match.group(1).strip() if bail_type_match else "N/A"
            law_description = law_description_match.group(1).strip() if law_description_match else "N/A"

            # Create the JSON response
            response = {
                "summary": summary,
                "bail_score": bail_score,
                "court_eligibility": court_eligibility,
                "bail_type": bail_type,
                "law_description": law_description
            }

            

            

            # base64_pdf = pdf_to_base64(pdf_file)
            # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)

            st.success("Analysis Complete!")
            st.write(response)

            st.download_button(
            label="Download PDF of bail application",
            data= generate_pdf_report(
                summary=response["summary"],
                bail_score=response["bail_score"],
                court_eligibility=response["court_eligibility"],
                bail_type=response["bail_type"],
                law_description=response["law_description"]
            ),
            file_name="bail_application_report.pdf",
            mime="application/pdf"
        )
            st.download_button(
            label="Download PDF of analysis",
            data= generate_pdf(
                legal_response
            ),
            file_name="analysis_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
