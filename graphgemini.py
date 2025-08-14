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

def generate_html_org_chart(hierarchy_data):
    """
    Generate an HTML-based organizational chart from the hierarchy data.
    Creates role-based nodes with people inside, arranged hierarchically.
    """
    # Group people by role
    role_groups = {}
    for person, data in hierarchy_data.items():
        role = data["role"]
        if role not in role_groups:
            role_groups[role] = []
        role_groups[role].append(person)
    
    # Build role hierarchy (which role reports to which role)
    role_hierarchy = {}
    for person, data in hierarchy_data.items():
        role = data["role"]
        reports_to_person = data["reports_to"]
        
        if reports_to_person:
            # Find the role of the person they report to
            reports_to_role = hierarchy_data[reports_to_person]["role"]
            role_hierarchy[role] = reports_to_role
        else:
            role_hierarchy[role] = None
    
    # Find root roles (roles with no parent)
    root_roles = [role for role, parent in role_hierarchy.items() if parent is None]
    
    # Build the HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Organizational Chart</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .org-chart {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 40px;
        }}
        
        .level {{
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .node {{
            background: white;
            border: 2px solid #4a90e2;
            border-radius: 12px;
            padding: 15px;
            min-width: 200px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .node:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.2);
        }}
        
        .node.role-node {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #4a90e2;
        }}
        
        .node.role-node .role-title {{
            font-weight: bold;
            font-size: 16px;
            text-align: center;
            margin-bottom: 10px;
            padding: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
        }}
        
        .people-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .person {{
            background: rgba(255,255,255,0.9);
            color: #333;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            text-align: center;
            border-left: 4px solid #4a90e2;
        }}
        
        .connection-line {{
            position: absolute;
            background: #4a90e2;
            width: 2px;
            height: 20px;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
        }}
        
        .connection-line.horizontal {{
            width: 30px;
            height: 2px;
            bottom: auto;
            top: 50%;
            transform: translateY(-50%);
        }}
        
        .connection-line.horizontal.left {{
            right: 100%;
            left: auto;
        }}
        
        .connection-line.horizontal.right {{
            left: 100%;
            right: auto;
        }}
        
        .level-label {{
            color: white;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .chart-title {{
            color: white;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 40px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .stats {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            color: white;
            text-align: center;
        }}
        
        .stats h3 {{
            margin: 0 0 15px 0;
            font-size: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 10px;
            border-radius: 8px;
        }}
        
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #ffd700;
        }}
        
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="chart-title">🏢 Organizational Chart</div>
    
    <div class="stats">
        <h3>📊 Organization Statistics</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{len(hierarchy_data)}</div>
                <div class="stat-label">Total People</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(role_groups)}</div>
                <div class="stat-label">Unique Roles</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(root_roles)}</div>
                <div class="stat-label">Top Level Roles</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{max(len(people) for people in role_groups.values()) if role_groups else 0}</div>
                <div class="stat-label">Largest Team</div>
            </div>
        </div>
    </div>
    
    <div class="org-chart">
"""
    
    # Generate levels based on hierarchy depth
    levels = build_hierarchy_levels(role_hierarchy, root_roles)
    
    for level_num, level_roles in enumerate(levels):
        html_content += f'        <div class="level-label">Level {level_num + 1}</div>\n'
        html_content += '        <div class="level">\n'
        
        for role in level_roles:
            people = role_groups.get(role, [])
            html_content += f'            <div class="node role-node">\n'
            html_content += f'                <div class="role-title">{role}</div>\n'
            html_content += '                <div class="people-list">\n'
            
            for person in people:
                html_content += f'                    <div class="person">{person}</div>\n'
            
            html_content += '                </div>\n'
            
            # Add connection line if not at root level
            if level_num > 0:
                html_content += '                <div class="connection-line"></div>\n'
            
            html_content += '            </div>\n'
        
        html_content += '        </div>\n'
    
    html_content += """
    </div>
    
    <script>
        // Add some interactive features
        document.querySelectorAll('.node').forEach(node => {
            node.addEventListener('click', function() {
                this.style.transform = 'scale(1.05)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 200);
            });
        });
        
        // Add smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
"""
    
    return html_content

def build_hierarchy_levels(role_hierarchy, root_roles):
    """
    Build hierarchical levels based on role hierarchy.
    Returns a list of lists, where each inner list contains roles at that level.
    """
    levels = [root_roles]
    processed_roles = set(root_roles)
    
    while True:
        next_level = []
        for role in levels[-1]:
            # Find all roles that report to this role
            for child_role, parent_role in role_hierarchy.items():
                if parent_role == role and child_role not in processed_roles:
                    next_level.append(child_role)
                    processed_roles.add(child_role)
        
        if not next_level:
            break
        
        levels.append(next_level)
    
    return levels

def save_html_chart(hierarchy_data, filename="org_chart.html"):
    """
    Save the HTML organizational chart to a file.
    """
    html_content = generate_html_org_chart(hierarchy_data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

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
                
                # Generate and display HTML chart
                st.subheader("HTML Organizational Chart")
                html_content = generate_html_org_chart(full_hierarchy_data)
                
                # Display HTML in an iframe or provide download
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button for HTML file
                st.download_button(
                    label="⬇️ Download HTML Chart",
                    data=html_content,
                    file_name="organizational_chart.html",
                    mime="text/html"
                )
                
                st.markdown("---")
                st.subheader("Inferred Hierarchy Data (JSON)")
                st.json(full_hierarchy_data)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error("Please check your file format and data columns.")
