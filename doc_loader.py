# document_loader.py
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests

def extract_doc_id(url):
    """Extract document ID from Google Docs URL"""
    if '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    return url

def fetch_google_doc(doc_id):
    """Fetch document content from public Google Doc"""
    # For public docs, use the export API
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    
    if response.status_code == 200:
        return response.text
    else:
        # Fallback to API method
        SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
        creds = service_account.Credentials.from_service_account_file(
            'credentials.json', scopes=SCOPES)
        
        service = build('docs', 'v1', credentials=creds)
        document = service.documents().get(documentId=doc_id).execute()
        
        return extract_text_from_doc(document)

def extract_text_from_doc(document):
    """Parse Google Docs JSON structure"""
    content = document.get('body').get('content', [])
    text_parts = []
    
    for element in content:
        if 'paragraph' in element:
            for text_elem in element['paragraph']['elements']:
                if 'textRun' in text_elem:
                    text_parts.append(text_elem['textRun']['content'])
    
    return ''.join(text_parts)
