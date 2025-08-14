import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import graphviz
import json
import io

# Load the environment variables from the .env file
load_dotenv()

# --- Main App Configuration ---
# Set the title and icon for the Streamlit page
st.set_page_config(
    page_title="AI-Powered Org Chart Generator",
    page_icon="📊",
    layout="wide"
)

# Load OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_hierarchy_from_ai(df):
    """
    Calls the OpenAI API to infer the organizational hierarchy from a DataFrame.
    The prompt is carefully crafted to request a JSON output that can be easily parsed.
    """
    st.info("AI is analyzing the roles to build the hierarchy...")
    
    # Create a simple, clean string representation of the DataFrame for the prompt
    data_string = df.to_csv(index=False)
    
    prompt = f"""
    You are an expert at inferring organizational hierarchy from a list of names and roles.
    Based on the following CSV data, identify the reporting structure.
    A person reports to another if their role is less senior (e.g., 'Electrician' reports to 'Ele Chargehand', 'Ele Chargehand' reports to 'Ele Supervisor').
    The data is as follows:
    ---
    {data_string}
    ---
    
    Please provide the output as a JSON object where the key is the person's name and the value is an object containing their 'role' and the 'reports_to' field. The 'reports_to' field should contain the name of their direct manager. If a person is at the top of the hierarchy (e.g., a supervisor), set 'reports_to' to null.
    
    Example format:
    {{
      "Ramesh Kumar Krishnappa Bangera": {{
        "role": "Ele Supervisor",
        "reports_to": null
      }},
      "Murugan Karuppaiyan Karuppaiyan": {{
        "role": "Ele Chargehand",
        "reports_to": "Ramesh Kumar Krishnappa Bangera"
      }},
      "SAMSE ALAM KHAN": {{
        "role": "Electrician",
        "reports_to": "Murugan Karuppaiyan Karuppaiyan"
      }}
    }}
    
    Strictly provide only the JSON object, do not include any other text or explanation.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Using a powerful model for better inference
            messages=[
                {"role": "system", "content": "You are a helpful assistant that infers organizational charts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Use a low temperature for consistent, predictable output
        )
        
        # Extract the JSON string from the response
        json_string = response.choices[0].message.content
        st.success("Hierarchy inferred successfully!")
        
        # --- FIX: Added a try-except block to handle potential JSON parsing errors gracefully ---
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as json_e:
            st.error("The AI's response was not in the expected JSON format. Please check the API response for errors.")
            st.code(json_string, language="json")
            return None
            
    except Exception as e:
        st.error(f"An error occurred while calling the AI: {e}")
        st.error("Please ensure your OpenAI API key is correct and that the model is available.")
        return None

def draw_org_chart(hierarchy_data):
    """
    Draws the organizational chart using the graphviz library.
    """
    dot = graphviz.Digraph(comment="Organizational Chart", graph_attr={'rankdir': 'TB'})
    
    # Add nodes (people) to the graph
    for name, data in hierarchy_data.items():
        role = data.get("role", "Unknown Role")
        
        # Create a visually appealing node with the person's name and role
        node_label = f"<{name}<br/><font point-size='10'>{role}</font>>"
        dot.node(name, label=node_label, shape="box", style="rounded,filled", fillcolor="lightgray", fontname="Helvetica")
        
        # Add edges (reporting lines)
        reports_to = data.get("reports_to")
        if reports_to:
            dot.edge(reports_to, name)
            
    return dot

# --- Streamlit UI Layout ---
st.title("Organizational Chart Generator 📊")

st.markdown("""
Upload your employee data (CSV or Excel) and this app will use AI to automatically
infer the reporting hierarchy and draw a professional organizational chart.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Use a spinner while processing
    with st.spinner('Reading and processing file...'):
        try:
            # Read the file based on its extension
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file)
            elif file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                st.stop()

            # Display the raw data for user verification
            st.subheader("Uploaded Data Preview")
            st.dataframe(df)

            # Get the hierarchy from the AI
            hierarchy_data = get_hierarchy_from_ai(df)

            if hierarchy_data:
                st.subheader("Generated Organizational Chart")
                
                # Draw and display the chart
                org_chart = draw_org_chart(hierarchy_data)
                st.graphviz_chart(org_chart)
                
                st.markdown("---")
                st.subheader("Inferred Hierarchy Data (JSON)")
                st.json(hierarchy_data)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error("Please check your file format and data columns.")
