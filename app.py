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
API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
if not API_KEY:
    st.info("Set your OpenAI key in a `.env` file as `OPENAI_API_KEY=...` (or in Streamlit Secrets).", icon="🔑")

os.environ["OPENAI_API_KEY"] = API_KEY or ""
client = OpenAI() if API_KEY else None

# ---------- PROMPTS ----------
SYSTEM_PROMPT = """You are a meticulous data labeling expert.
Your task: from a tiny sample of an Excel sheet (top values per column), decide:
1) Is this sheet relevant for extracting a list of people (humans) and their job roles?
2) Exactly which column(s) are "name" columns (people’s names).
3) Exactly which column(s) are "role" columns (job titles/roles).

CRITICAL RULES:
- Base your decision ONLY on the provided headers and sample values (not on assumptions).
- Consider multilingual contexts (e.g., Arabic, Hindi/Urdu, English).
- Names: typically proper names (often two words), may include initials, honorifics; avoid company/department names.
- Roles: words like Engineer, Supervisor, Foreman, Manager, Electrician, Mason, Architect, Carpenter, Laborer, Operator, Technician, Driver, etc. Also accept synonyms like 'Job Role', 'Designation', 'Position', 'Title'.
- EXCLUDE columns clearly not names/roles: ids, employee codes, phone/email, department, company, vendor, site name, dates, counts, rates, remarks/notes.
- A sheet is RELEVANT only if it's plausible to contain human names AND associated roles (job titles) in some row alignment.
- If there are multiple candidate name columns (e.g., First Name, Last Name), list them all under name_columns.
- If there are multiple role columns (e.g., Role + Sub-role), list them all under role_columns.
- If not relevant, set name_columns and role_columns to [] and is_relevant=false.
- Respond strictly as JSON in the schema below.

Return JSON in EXACT schema:
{
  "sheet_name": "<string>",
  "is_relevant": <true|false>,
  "name_columns": ["<exact header from sample>", ...],
  "role_columns": ["<exact header from sample>", ...],
  "reason": "<one concise sentence>"
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
            "instruction": "Classify which columns are person names and which are job roles.",
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

# ---------- HELPERS ----------
def read_excel_all_sheets(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    data = {}
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            data[sheet] = df
        except Exception:
            # Skip sheets that fail to parse
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
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: mark not relevant if JSON fails
        data = {
            "sheet_name": sheet_name,
            "is_relevant": False,
            "name_columns": [],
            "role_columns": [],
            "reason": "Model returned invalid JSON."
        }
    # normalize fields
    data.setdefault("sheet_name", sheet_name)
    data.setdefault("is_relevant", False)
    data.setdefault("name_columns", [])
    data.setdefault("role_columns", [])
    data.setdefault("reason", "")
    return data

def pick_first_nonempty(values: List[Any]) -> str:
    for v in values:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            return s
    return ""

def assemble_name(row: pd.Series, name_cols: List[str]) -> str:
    parts = [str(row[c]).strip() for c in name_cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
    # combine first/last if present
    return " ".join(parts).strip()

def assemble_role(row: pd.Series, role_cols: List[str]) -> str:
    parts = [str(row[c]).strip() for c in role_cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
    # Prefer first non-empty if they’re redundant; else join
    return parts[0] if len(parts) == 1 else " / ".join(parts)

def looks_like_person_name(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # basic heuristic to filter obvious non-names
    bad_tokens = ["company", "llc", "l.l.c", "ltd", "pvt", "private", "department", "unit", "contract", "scope"]
    if any(bt in s.lower() for bt in bad_tokens):
        return False
    # names often have letters and at least one space
    letters = sum(ch.isalpha() for ch in s)
    return letters >= 3

def looks_like_role(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # common construction / generic titles
    maybe_roles = [
        "engineer","supervisor","foreman","manager","mason","carpenter","electrician","plumber","technician",
        "operator","driver","architect","qa/qc","qs","draftsman","safety","helper","labor","mechanic",
        "site","civil","steel fixer","welder","painter","inspector","coordinator","administrator","secretary",
        "project","structural","mechanical","electrical","instrumentation","hvac","survey","storekeeper"
    ]
    return any(tok in s.lower() for tok in maybe_roles)

def extract_from_sheet(df: pd.DataFrame, name_cols: List[str], role_cols: List[str]) -> pd.DataFrame:
    subset_cols = [c for c in name_cols + role_cols if c in df.columns]
    if not subset_cols:
        return pd.DataFrame(columns=["Name", "Role"])

    out_rows = []
    for _, row in df[subset_cols].iterrows():
        name = assemble_name(row, name_cols) if name_cols else ""
        role = assemble_role(row, role_cols) if role_cols else ""
        if name and role and looks_like_person_name(name) and looks_like_role(role):
            out_rows.append({"Name": name, "Role": role})

    return pd.DataFrame(out_rows)

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
    st.markdown("---")
    st.caption("Tip: To reduce token usage, keep columns/rows sampled modest.")

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])

# Helper for local testing with a bundled file (optional)
# Uncomment if you want a "Use sample file" button:
# sample_path = "VIDA RESIDENCES-P110_MANPOWER LIST.xlsx"
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
        chunk = extract_from_sheet(df, name_cols, role_cols)
        if not chunk.empty:
            # attach sheet for traceability
            chunk["__sheet__"] = sheet_name
            extracted_chunks.append(chunk)

    if show_sheet_logs:
        with st.expander(f"🔎 {sheet_name} — Relevance: {classification.get('is_relevant')}"):
            st.json(classification)
            st.write("Sample seen by AI (first few values per column):")
            st.write({k: v["top_values"][:5] for k, v in sample.items()})

    progress.progress(i / len(all_sheets))

merged = merge_and_clean(extracted_chunks)

st.subheader("✅ Consolidated People (Name & Role)")
if merged.empty:
    st.warning("No rows extracted. Either the workbook lacks name/role info or the sampling missed them. Try increasing rows/columns sampled.")
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
