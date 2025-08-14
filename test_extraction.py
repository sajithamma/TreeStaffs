#!/usr/bin/env python3
"""
Test script to manually test the extraction function
"""
import pandas as pd
import openpyxl

def test_extraction():
    print("🧪 Testing extraction function manually...")
    
    # Test with the sample Excel file
    file_path = "sample.xlsx"
    
    try:
        # Read the Excel file
        print(f"📁 Reading file: {file_path}")
        xl = pd.ExcelFile(file_path)
        print(f"📊 Found sheets: {xl.sheet_names}")
        
        # Test with "Worker Details" sheet (second sheet)
        sheet_name = "Worker Details"
        print(f"\n🔍 Testing sheet: {sheet_name}")
        
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 DataFrame columns: {list(df.columns)}")
        print(f"📄 First 3 rows:")
        print(df.head(3))
        
        # Manually specify the columns (as identified by AI)
        name_cols = ["NAME"]
        role_cols = ["Job Role"]
        
        print(f"\n🎯 Testing extraction with:")
        print(f"   Name columns: {name_cols}")
        print(f"   Role columns: {role_cols}")
        
        # Test the extraction function
        result = extract_from_sheet(df, name_cols, role_cols)
        
        print(f"\n✅ Extraction result:")
        print(f"   Result shape: {result.shape}")
        if not result.empty:
            print(f"   First 5 rows:")
            print(result.head())
        else:
            print("   ❌ No data extracted!")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def extract_from_sheet(df: pd.DataFrame, name_cols: list, role_cols: list) -> pd.DataFrame:
    """
    Simple extraction: get all data from name and role columns, combine into a single dataframe
    """
    print(f"\n🔧 Inside extract_from_sheet function:")
    
    subset_cols = [c for c in name_cols + role_cols if c in df.columns]
    print(f"   Subset columns found: {subset_cols}")
    
    if not subset_cols:
        print(f"   ❌ No subset columns found!")
        print(f"   Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=["Name", "Role"])

    # Extract the columns we need
    extracted_data = df[subset_cols].copy()
    print(f"   📊 Extracted data shape: {extracted_data.shape}")
    print(f"   📊 Sample extracted data:")
    print(extracted_data.head(3))
    
    # Combine name columns into one "Name" column
    if name_cols:
        print(f"   🔗 Combining name columns: {name_cols}")
        extracted_data["Name"] = extracted_data[name_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        print(f"   📝 Sample names after combination:")
        print(extracted_data["Name"].head(3))
    
    # Combine role columns into one "Role" column  
    if role_cols:
        print(f"   🔗 Combining role columns: {role_cols}")
        extracted_data["Role"] = extracted_data[role_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        print(f"   📝 Sample roles after combination:")
        print(extracted_data["Role"].head(3))
    
    # Keep only Name and Role columns, remove empty rows
    result = extracted_data[["Name", "Role"]].copy()
    print(f"   🎯 Before filtering - result shape: {result.shape}")
    
    # Show some sample data before filtering
    print(f"   📊 Sample data before filtering:")
    print(result.head(3))
    
    # Filter out empty rows
    result = result[(result["Name"] != "") & (result["Role"] != "")]
    print(f"   🎯 After filtering - result shape: {result.shape}")
    
    if not result.empty:
        print(f"   ✅ Final result sample:")
        print(result.head(3))
    else:
        print(f"   ❌ All rows filtered out!")
        print(f"   📊 Checking what got filtered:")
        print(f"      Empty names: {(result['Name'] == '').sum()}")
        print(f"      Empty roles: {(result['Role'] == '').sum()}")
        print(f"      Sample names: {result['Name'].head(3).tolist()}")
        print(f"      Sample roles: {result['Role'].head(3).tolist()}")
    
    return result

if __name__ == "__main__":
    test_extraction()
