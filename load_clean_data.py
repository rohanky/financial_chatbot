import os
import pandas as pd
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

class DocumentReader:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.data = pd.DataFrame(columns=['Company', 'File Name', 'File Path', 'File Type', 'Page Number', 'Element Number', 'Content'])

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file with page and element numbers."""
        extracted_data = []
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page_number, page in enumerate(reader.pages, start=1):
                text_elements = page.extract_text().split("\n") if page.extract_text() else []
                for element_number, text in enumerate(text_elements, start=1):
                    extracted_data.append({
                        'Page Number': page_number,
                        'Element Number': element_number,
                        'Content': text
                    })
        return extracted_data

    def extract_text_from_html(self, file_path):
        """Extract text from an HTML file with element numbers and handle encoding issues."""
        extracted_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
        except UnicodeDecodeError:
            # Retry with a different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin1') as f:
                soup = BeautifulSoup(f, 'html.parser')

        text_elements = soup.stripped_strings  # Extract visible text without tags
        for element_number, text in enumerate(text_elements, start=1):
            extracted_data.append({
                'Page Number': None,  # HTML doesn't have pages
                'Element Number': element_number,
                'Content': text
            })
        return extracted_data

    def read_all_files(self):
        """Read all PDF and HTML files and store their content in a DataFrame."""
        file_data = []

        for company in os.listdir(self.data_path):
            company_path = os.path.join(self.data_path, company)
            if os.path.isdir(company_path):
                for file in os.listdir(company_path):
                    file_path = os.path.join(company_path, file)
                    file_type = file.split('.')[-1]

                    if file_type == 'pdf':
                        extracted_content = self.extract_text_from_pdf(file_path)
                    elif file_type == 'html':
                        extracted_content = self.extract_text_from_html(file_path)
                    else:
                        continue
                    
                    # Append each element's content with its metadata
                    for content_info in extracted_content:
                        file_data.append({
                            'Company': company,
                            'File Name': file,
                            'File Path': file_path,
                            'File Type': file_type,
                            'Page Number': content_info['Page Number'],
                            'Element Number': content_info['Element Number'],
                            'Content': content_info['Content']
                        })

        self.data = pd.DataFrame(file_data)

    def get_data(self):
        """Return the extracted data as a DataFrame."""
        return self.data


# Example Usage
document_reader = DocumentReader(data_path='data/')
document_reader.read_all_files()

# Get the DataFrame
df = document_reader.get_data()
print(df.head())  # Display the first few rows

# Optionally save the data to a CSV file
df.to_excel('parsed_data.xlsx')