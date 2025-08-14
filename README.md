# Multi-Sheet Name & Role Extractor

A Streamlit application that uses OpenAI's GPT-4o to intelligently extract names and job roles from multi-sheet Excel files, specifically designed for construction and manpower management data.

## Features

- 🔍 **Intelligent Column Detection**: Uses AI to automatically identify name and role columns across different sheet formats
- 📊 **Multi-Sheet Processing**: Processes all sheets in an Excel workbook and extracts relevant data
- 🏗️ **Construction Industry Optimized**: Specialized prompts for construction job titles and workforce data
- 📥 **Flexible Output**: Download results as Excel (.xlsx) or CSV files
- 🎯 **Smart Filtering**: Automatically skips irrelevant sheets and validates extracted data
- 🌍 **Multilingual Support**: Handles names and roles in multiple languages

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

**Important**: Replace `your_actual_openai_api_key_here` with your real OpenAI API key.

### 3. Run the Application

```bash
streamlit run app.py
```

## How It Works

### 1. **Sheet Analysis**
- Reads all sheets from the uploaded Excel file
- Samples the first few rows of each column to understand the data structure
- Uses GPT-4o to analyze column headers and sample values

### 2. **AI Classification**
The AI determines for each sheet:
- Whether it contains relevant people/role data
- Which columns contain person names
- Which columns contain job roles/positions

### 3. **Data Extraction**
- Extracts name-role pairs from relevant sheets
- Combines data from multiple sheets into a single list
- Removes duplicates and validates data quality

### 4. **Output Generation**
- Creates a consolidated list of all people with their roles
- Provides download options for Excel and CSV formats
- Includes source sheet information for traceability

## Supported Data Formats

### Name Columns
- Full names (e.g., "RAMESH KUMAR KRISHNAPPA BANGERA")
- Names with initials
- First/Last name combinations
- Multilingual names (Arabic, Hindi, English, etc.)

### Role Columns
- **Construction Roles**: Engineer, Supervisor, Foreman, Mason, Carpenter
- **Technical Roles**: Electrician, Plumber, Technician, Operator
- **Management Roles**: Project Manager, Team Lead, Coordinator
- **Support Roles**: Document Controller, Safety Officer, Storekeeper

### Column Headers
The AI recognizes various column naming conventions:
- "Name", "NAME", "Employee Name"
- "Role", "Job Role", "Designation", "Position", "Title"

## Configuration Options

### Sampling Settings
- **Rows per column**: Number of sample rows the AI analyzes (3-25)
- **Max columns**: Maximum columns to process per sheet (5-50)

### Model Selection
- Currently supports GPT-4o for optimal performance

## Example Output

The application generates a consolidated Excel file with two columns:
- **Name**: Full name of the person
- **Role**: Job title/position

## Troubleshooting

### Common Issues

1. **"No rows extracted"**
   - Increase the number of rows sampled per column
   - Check if your Excel file has the expected structure
   - Ensure sheets contain both names and roles

2. **"OpenAI API key not set"**
   - Verify your `.env` file exists and contains the correct API key
   - Check that the API key is valid and has sufficient credits

3. **Sheets not being processed**
   - Some sheets may be automatically skipped if they don't contain relevant data
   - Check the expandable logs to see why sheets were skipped

### Performance Tips

- Keep sampling settings modest to reduce API token usage
- For large files, consider processing in batches
- The AI only sees sample data, not the entire file

## File Structure

```
TreeStaffs/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── sample.xlsx         # Sample Excel file for testing
├── .env               # Environment variables (create this)
└── README.md          # This documentation
```

## Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and Excel processing
- **OpenPyXL**: Excel file reading/writing
- **OpenAI**: AI-powered column classification
- **Python-dotenv**: Environment variable management

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the application logs for error details
3. Ensure your Excel file structure matches the expected format
