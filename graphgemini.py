import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import graphviz
import json
import io
import itertools

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

def get_role_hierarchy_from_ai(unique_roles):
    """
    Calls the OpenAI API to infer the organizational hierarchy of roles.
    This is a more stable approach as it sends a much smaller list to the AI.
    """
    st.info("AI is analyzing the unique roles to build the hierarchy structure...")
    
    # Create a clean string representation of the unique roles for the prompt
    roles_string = "\n".join(unique_roles)
    
    prompt = f"""
    You are an expert at inferring organizational hierarchy from a list of job roles.
    Based on the following unique roles, identify the reporting structure.
    A role reports to another if it is less senior. For example, 'Electrician' reports to 'Ele Chargehand', which reports to 'Ele Supervisor'.
    The list of unique roles is as follows:
    ---
    {roles_string}
    ---
    
    Please provide the output as a JSON object where the key is the role name and the value is the name of the role's direct manager. If a role is at the top of the hierarchy (e.g., a supervisor), set its value to null.
    
    Example format:
    {{
      "Ele Supervisor": null,
      "Ele Chargehand": "Ele Supervisor",
      "Electrician": "Ele Chargehand",
      "Ass Electrician": "Electrician"
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
        
        # Extract and parse the JSON string from the response
        json_string = response.choices[0].message.content
        st.success("Role hierarchy inferred successfully!")
        
        try:
            # --- FIX: Pre-process the string to remove markdown code block markers ---
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            
            return json.loads(json_string.strip())
        except json.JSONDecodeError as json_e:
            st.error("The AI's response was not in the expected JSON format. The raw output is displayed below for debugging.")
            st.code(json_string, language="text")
            return None
            
    except Exception as e:
        st.error(f"An error occurred while calling the AI: {e}")
        st.error("Please ensure your OpenAI API key is correct and that the model is available.")
        return None

def build_full_hierarchy(df, role_hierarchy):
    """
    Builds the full hierarchy of people and roles based on the inferred role hierarchy.
    """
    st.info("Mapping individuals to their roles in the hierarchy...")
    
    people_by_role = df.groupby('Role')['Name'].apply(list).to_dict()
    full_hierarchy = {}

    for role, reports_to_role in role_hierarchy.items():
        # Get the list of people for the current role
        people_in_role = people_by_role.get(role, [])
        
        # Find the managers for this role
        managers = people_by_role.get(reports_to_role, []) if reports_to_role else [None]

        # Handle the case where no manager is found
        if not managers:
            managers = [None]
        
        # Cycle through managers to distribute direct reports
        manager_cycle = itertools.cycle(managers)

        # Assign each person to a manager
        for person in people_in_role:
            manager = next(manager_cycle)
            full_hierarchy[person] = {
                "role": role,
                "reports_to": manager
            }
    
    st.success("Full hierarchy successfully built!")
    return full_hierarchy


def draw_org_chart(hierarchy_data):
    """
    Draws the organizational chart using the graphviz library.
    """
    dot = graphviz.Digraph(comment="Organizational Chart", graph_attr={'rankdir': 'TB'})
    
    # Add nodes (people) and edges (reporting lines) to the graph
    for name, data in hierarchy_data.items():
        role = data.get("role", "Unknown Role")
        
        # Create a visually appealing node
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
This version first determines the hierarchy of roles (a more stable approach) and then
maps people to those roles.
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

            # Ensure the required columns exist
            if 'Name' not in df.columns or 'Role' not in df.columns:
                st.error("The uploaded file must contain 'Name' and 'Role' columns.")
                st.stop()
            
            # Display the raw data for user verification
            st.subheader("Uploaded Data Preview")
            st.dataframe(df)

            # Get unique roles from the dataframe
            unique_roles = df['Role'].unique().tolist()
            
            # Get the role hierarchy from the AI
            role_hierarchy = get_role_hierarchy_from_ai(unique_roles)
            
            if role_hierarchy:
                # Build the full hierarchy of people based on the role hierarchy
                full_hierarchy_data = build_full_hierarchy(df, role_hierarchy)
                
                st.subheader("Generated Organizational Chart")
                
                # Draw and display the chart
                org_chart = draw_org_chart(full_hierarchy_data)
                st.graphviz_chart(org_chart)
                
                st.markdown("---")
                st.subheader("Inferred Hierarchy Data (JSON)")
                st.json(full_hierarchy_data)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error("Please check your file format and data columns.")
