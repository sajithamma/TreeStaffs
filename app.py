# app.py
import os
import io
import json
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

# pip install python-dotenv openai pandas streamlit openpyxl
from dotenv import load_dotenv
from openai import OpenAI

# ---------- STREAMLIT BASICS ----------
st.set_page_config(page_title="Name & Role Extractor", layout="wide")

st.title("🧾 Multi-sheet Name & Role Extractor")
st.caption("Uploads an Excel with many tabs, uses GPT-4o to detect Name & Role columns, merges into a single list, and lets you download the result.")

# ---------- ENV / OPENAI CLIENT ----------
load_dotenv()  # loads .env if present

# You can set OPENAI_API_KEY in either .env or Streamlit Secrets
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    # Try to get from Streamlit secrets if available
    try:
        API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    except FileNotFoundError:
        # No secrets file found, continue with empty API key
        pass

if not API_KEY:
    st.info("Set your OpenAI key in a `.env` file as `OPENAI_API_KEY=...` (or in Streamlit Secrets).", icon="🔑")

if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY
    client = OpenAI()
else:
    client = None

# ---------- IMPROVED PROMPTS ----------
SYSTEM_PROMPT = """You are a specialized data analyst for construction and manpower management.
Your task: from a sample of an Excel sheet (headers + top values per column), determine:
1) Is this sheet relevant for extracting a list of people (humans) and their job roles?
2) Exactly which column(s) contain person names.
3) Exactly which column(s) contain job roles/positions.

CONSTRUCTION INDUSTRY CONTEXT:
- This is typically for construction projects, manpower planning, or workforce management
- Names: Employee names, worker names, staff names (often full names with initials)
- Roles: Construction job titles like Engineer, Supervisor, Foreman, Mason, Carpenter, Electrician, etc.

CRITICAL RULES:
- Base decisions ONLY on provided headers and sample values
- Consider multilingual contexts (Arabic, Hindi/Urdu, English, etc.)
- Names: Look for columns with proper names (2+ words), may include initials, honorifics
- Roles: Look for job titles, designations, positions (e.g., 'Job Role', 'Role', 'Designation', 'Position', 'Title')
- EXCLUDE: IDs, codes, phone/email, department, company, vendor, site name, dates, counts, rates, remarks
- A sheet is RELEVANT if it plausibly contains human names AND associated roles in aligned rows
- Handle multiple name columns (e.g., First Name + Last Name) by listing all under name_columns
- Handle multiple role columns by listing all under role_columns
- If not relevant, set name_columns and role_columns to [] and is_relevant=false

Return JSON in EXACT schema:
{
  "sheet_name": "<string>",
  "is_relevant": <true|false>,
  "name_columns": ["<exact header from sample>", ...],
  "role_columns": ["<exact header from sample>", ...],
  "reason": "<one concise sentence explaining the decision>"
}
"""

def build_user_prompt(sheet_name: str, sample: Dict[str, Any]) -> str:
    """
    We pass a compact snapshot of the sheet:
    - the sheet name
    - for each column: header + up to N top non-empty values
    """
    return json.dumps(
        {
            "instruction": "Classify which columns are person names and which are job roles for construction/manpower data.",
            "sheet_name": sheet_name,
            "columns": [
                {
                    "header": str(col),
                    "top_values": sample[str(col)]["top_values"]
                }
                for col in sample.keys()
            ],
        },
        ensure_ascii=False
    )

# ---------- IMPROVED HELPERS ----------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataframe to handle common Excel issues.
    """
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Clean column names - remove extra spaces and normalize
    df.columns = [str(col).strip() for col in df.columns]
    
    # Try to find the first row with actual data (skip header rows)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        if any(str(val).strip() and str(val).lower() not in ['nan', 'none', ''] for val in row):
            # This row has data, use it as starting point
            df = df.iloc[i:].reset_index(drop=True)
            break
    
    return df

def read_excel_all_sheets(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    data = {}
    for sheet in xls.sheet_names:
        try:
            # First try to read with header to preserve column names
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                if st.session_state.get('debug_mode', False):
                    st.write(f"📊 **Sheet '{sheet}' loaded with header:**")
                    st.write(f"   Shape: {df.shape}")
                    st.write(f"   Columns: {list(df.columns)}")
                    st.write(f"   First 3 rows:")
                    st.write(df.head(3))
            except:
                # Fallback to no header if that fails
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
                df = clean_dataframe(df)
                if st.session_state.get('debug_mode', False):
                    st.write(f"📊 **Sheet '{sheet}' loaded without header:**")
                    st.write(f"   Shape: {df.shape}")
                    st.write(f"   Columns: {list(df.columns)}")
                    st.write(f"   First 3 rows:")
                    st.write(df.head(3))
            
            if not df.empty and len(df.columns) >= 2:
                data[sheet] = df
        except Exception as e:
            st.warning(f"Could not read sheet '{sheet}': {str(e)}")
            continue
    return data

def sample_sheet(df: pd.DataFrame, max_rows: int = 10, max_cols: int = 25) -> Dict[str, Dict[str, List[str]]]:
    # limit columns (sometimes files have dozens of helper columns)
    cols = list(map(str, df.columns[:max_cols]))
    sample = {}
    for c in cols:
        # pick first max_rows non-null, meaningful strings
        vals = df[c].dropna()
        # convert to strings and strip
        vals = vals.astype(str).map(lambda x: x.strip()).replace({"": None}).dropna()
        top_vals = vals.head(max_rows).tolist()
        sample[str(c)] = {"top_values": top_vals}
    return sample

def classify_columns_with_gpt(sheet_name: str, sample: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Calls GPT-4o to decide relevancy + which columns are names vs roles.
    """
    if not client:
        # If no API, return a safe default "not relevant"
        return {
            "sheet_name": sheet_name,
            "is_relevant": False,
            "name_columns": [],
            "role_columns": [],
            "reason": "OpenAI API key not set; skipping."
        }

    user_prompt = build_user_prompt(sheet_name, sample)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: mark not relevant if JSON fails or API call fails
        data = {
            "sheet_name": sheet_name,
            "is_relevant": False,
            "name_columns": [],
            "role_columns": [],
            "reason": f"Error processing sheet: {str(e)}"
        }
    
    # normalize fields
    data.setdefault("sheet_name", sheet_name)
    data.setdefault("is_relevant", False)
    data.setdefault("name_columns", [])
    data.setdefault("role_columns", [])
    data.setdefault("reason", "")
    return data

# Removed unused helper functions - simplified extraction approach

# Removed complex validation functions - we now extract all data directly

def extract_from_sheet(df: pd.DataFrame, name_cols: List[str], role_cols: List[str]) -> pd.DataFrame:
    """
    Simple extraction: get all data from name and role columns, combine into a single dataframe
    """
    subset_cols = [c for c in name_cols + role_cols if c in df.columns]
    if st.session_state.get('debug_mode', False):
        st.write(f"🔍 **Column matching debug:**")
        st.write(f"   AI identified name_cols: {name_cols}")
        st.write(f"   AI identified role_cols: {role_cols}")
        st.write(f"   DataFrame columns: {list(df.columns)}")
        st.write(f"   Subset columns found: {subset_cols}")
        st.write(f"   Missing columns: {[c for c in name_cols + role_cols if c not in df.columns]}")
    
    if not subset_cols:
        if st.session_state.get('debug_mode', False):
            st.write(f"❌ No subset columns found. name_cols: {name_cols}, role_cols: {role_cols}")
            st.write(f"Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=["Name", "Role"])

    if st.session_state.get('debug_mode', False):
        st.write(f"✅ Found subset columns: {subset_cols}")

    # Extract the columns we need
    extracted_data = df[subset_cols].copy()
    
    if st.session_state.get('debug_mode', False):
        st.write(f"📊 Extracted data shape: {extracted_data.shape}")
        st.write(f"📊 Sample extracted data:")
        st.write(extracted_data.head(3))
    
    # Combine name columns into one "Name" column
    if name_cols:
        extracted_data["Name"] = extracted_data[name_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    
    # Combine role columns into one "Role" column  
    if role_cols:
        extracted_data["Role"] = extracted_data[role_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    
    # Keep only Name and Role columns, remove empty rows
    result = extracted_data[["Name", "Role"]].copy()
    result = result[(result["Name"] != "") & (result["Role"] != "")]
    
    if st.session_state.get('debug_mode', False):
        st.write(f"🎯 Final result shape: {result.shape}")
        if not result.empty:
            st.write("🎯 Sample final result:")
            st.write(result.head(3))
    
    return result

def merge_and_clean(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["Name", "Role"])
    merged = pd.concat(dfs, ignore_index=True)
    # remove duplicates & empties
    merged["Name"] = merged["Name"].astype(str).str.strip()
    merged["Role"] = merged["Role"].astype(str).str.strip()
    merged = merged[(merged["Name"] != "") & (merged["Role"] != "")]
    merged = merged.drop_duplicates(subset=["Name", "Role"]).reset_index(drop=True)
    return merged

# ---------- SIDEBAR OPTIONS ----------
with st.sidebar:
    st.header("Settings")
    max_rows = st.slider("Rows to sample per column", 3, 25, 10, help="AI sees only this many top non-empty rows per column.")
    max_cols = st.slider("Max columns to sample per sheet", 5, 50, 25, help="Caps the number of columns the AI sees.")
    model = st.selectbox("OpenAI Model", ["gpt-4o"], index=0)
    show_sheet_logs = st.checkbox("Show per-sheet classification logs", value=True)
    debug_mode = st.checkbox("Debug mode (show extraction details)", value=False)
    st.session_state['debug_mode'] = debug_mode
    st.markdown("---")
    st.caption("Tip: To reduce token usage, keep columns/rows sampled modest.")

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])

# Helper for local testing with a bundled file (optional)
# Uncomment if you want a "Use sample file" button:
# sample_path = "sample.xlsx"
# use_sample = st.button("Use sample file in app directory")

if uploaded is None:
    st.stop()

file_bytes = uploaded.read()
all_sheets = read_excel_all_sheets(file_bytes)
if not all_sheets:
    st.error("Could not read any sheets from the workbook.")
    st.stop()

st.success(f"Loaded {len(all_sheets)} sheets.")
progress = st.progress(0)
sheet_results: List[Tuple[str, Dict[str, Any]]] = []
extracted_chunks: List[pd.DataFrame] = []

for i, (sheet_name, df) in enumerate(all_sheets.items(), start=1):
    # skip trivially empty sheets
    if df.empty or len(df.columns) < 2:
        if show_sheet_logs:
            with st.expander(f"🗂️ {sheet_name} (skipped: empty/too few columns)"):
                st.write(df.head(5))
        progress.progress(i / len(all_sheets))
        continue

    sample = sample_sheet(df, max_rows=max_rows, max_cols=max_cols)
    classification = classify_columns_with_gpt(sheet_name, sample, model=model)

    sheet_results.append((sheet_name, classification))

    if classification.get("is_relevant") and classification.get("name_columns") and classification.get("role_columns"):
        name_cols = classification["name_columns"]
        role_cols = classification["role_columns"]
        
        # Always show what AI identified (simple and clear)
        st.write(f"📋 **{sheet_name}:** AI identified Name columns: `{name_cols}` | Role columns: `{role_cols}`")
        
        if debug_mode:
            st.write(f"🔍 **Debug for {sheet_name}:**")
            st.write(f"DataFrame columns: {list(df.columns)}")
            st.write(f"DataFrame shape: {df.shape}")
            st.write(f"DataFrame info:")
            st.write(df.info())
            st.write(f"First few rows of data:")
            st.write(df.head(5))
            st.write(f"Column types:")
            st.write(df.dtypes)
            st.write(f"Checking if columns exist:")
            for col in name_cols + role_cols:
                exists = col in df.columns
                st.write(f"   '{col}' exists: {exists}")
                if exists:
                    st.write(f"   '{col}' sample values: {df[col].head(3).tolist()}")
                else:
                    st.write(f"   Similar columns: {[c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]}")
        
        chunk = extract_from_sheet(df, name_cols, role_cols)
        
        if debug_mode:
            st.write(f"Extracted {len(chunk)} rows from {sheet_name}")
            if not chunk.empty:
                st.write("Sample extracted data:")
                st.write(chunk.head(3))
            else:
                st.write("⚠️ **No data extracted!** Let's see why:")
                st.write(f"DataFrame shape: {df.shape}")
                st.write(f"Sample data from name columns: {df[name_cols].head(3) if name_cols else 'No name columns'}")
                st.write(f"Sample data from role columns: {df[role_cols].head(3) if role_cols else 'No role columns'}")
        
        if not chunk.empty:
            # attach sheet for traceability
            chunk["__sheet__"] = sheet_name
            extracted_chunks.append(chunk)
            st.write(f"✅ **Successfully extracted {len(chunk)} rows from {sheet_name}**")
        else:
            st.write(f"❌ **No data extracted from {sheet_name}**")

    if show_sheet_logs:
        with st.expander(f"🔎 {sheet_name} — Relevance: {classification.get('is_relevant')}"):
            st.json(classification)
            st.write("Sample seen by AI (first few values per column):")
            st.write({k: v["top_values"][:5] for k, v in sample.items()})

    progress.progress(i / len(all_sheets))

st.write(f"📊 **Extraction Summary:** Total chunks collected: {len(extracted_chunks)}")
for i, chunk in enumerate(extracted_chunks):
    st.write(f"   Chunk {i+1}: {len(chunk)} rows from sheet '{chunk['__sheet__'].iloc[0]}'")

merged = merge_and_clean(extracted_chunks)

st.subheader("✅ Consolidated People (Name & Role)")
if merged.empty:
    st.warning("No rows extracted. Either the workbook lacks name/role info or the sampling missed them. Try increasing rows/columns sampled.")
    st.write(f"🔍 **Debug info:** extracted_chunks length: {len(extracted_chunks)}")
    if extracted_chunks:
        st.write("First chunk sample:")
        st.write(extracted_chunks[0].head(3))
else:
    # Light editor so you can quickly fix typos before download
    edited = st.data_editor(
        merged[["Name", "Role", "__sheet__"]],
        use_container_width=True,
        num_rows="dynamic",
        key="editor"
    )

    # Download buttons
    def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Write a single sheet called 'People'
            df[["Name", "Role"]].to_excel(writer, sheet_name="People", index=False)
        buffer.seek(0)
        return buffer.read()

    xlsx_bytes = df_to_excel_bytes(edited)
    st.download_button(
        label="⬇️ Download as Excel (single sheet: People)",
        data=xlsx_bytes,
        file_name="people_name_role.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv_bytes = edited[["Name", "Role"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_bytes,
        file_name="people_name_role.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Built with Streamlit + pandas + OpenAI GPT-4o. The model only sees a small sample (top N values per column) to judge column semantics, then the app extracts full rows for selected columns.")
