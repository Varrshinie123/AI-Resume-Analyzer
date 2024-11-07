import os
import multiprocessing as mp
import io
import spacy
from spacy.matcher import Matcher
import pprint
import chardet
from PyPDF2 import PdfReader
import pdfplumber
from PyPDF2.errors import PdfReadError
import pytesseract
from pdf2image import convert_from_path
from PyPDF2.errors import PdfReadError



class ResumeParser(object):

    def __init__(self, resume, skills_file=None, custom_regex=None):
        nlp = spacy.load('en_core_web_sm')
        self.__matcher = Matcher(nlp.vocab)
        self.__details = {
            'name': None,
            'email': None,
            'mobile_number': None,
            'skills': None,
            'degree': None,
            'no_of_pages': None,
        }
        self.__resume = resume
        if not isinstance(self.__resume, io.BytesIO):
            ext = os.path.splitext(self.__resume)[1].split('.')[1]
        else:
            ext = self.__resume.name.split('.')[1]
        
        self.__text_raw = self.extract_text(self.__resume, '.' + ext)
        self.__text = ' '.join(self.__text_raw.split())
        self.__nlp = nlp(self.__text)
        self.__noun_chunks = list(self.__nlp.noun_chunks)
        self.__get_basic_details()

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        name = self.extract_name(self.__nlp)
        email = self.extract_email(self.__text)
        mobile = self.extract_mobile_number(self.__text)
        skills = self.extract_skills(self.__nlp, self.__noun_chunks)

        # Populate details
        self.__details['name'] = name
        self.__details['email'] = email
        self.__details['mobile_number'] = mobile
        self.__details['skills'] = skills
        self.__details['no_of_pages'] = self.get_number_of_pages(self.__resume)

    
    @staticmethod
    def extract_text(resume, ext):
        if ext == '.pdf':
            try:
                with pdfplumber.open(resume) as pdf:
                    text = ''.join(page.extract_text() or '' for page in pdf.pages)
                return text
            except PdfReadError:
                print("Could not read PDF; attempting OCR.")
                try:
                    # OCR fallback if pdfplumber fails
                    images = convert_from_path(resume)
                    text = ''.join(pytesseract.image_to_string(img) for img in images)
                    return text
                except Exception as e:
                    print(f"OCR failed: {e}")
                    return ""
        else:
            # Handle text or other file types
            with open(resume, 'rb') as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']
                
                try:
                    return raw_data.decode(detected_encoding if detected_encoding else 'utf-8')
                except UnicodeDecodeError:
                    return raw_data.decode('ISO-8859-1', errors='ignore')
    @staticmethod
    def extract_name(nlp_doc):
        for ent in nlp_doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    @staticmethod
    def extract_email(text):
        import re
        match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        return match.group(0) if match else None

    @staticmethod
    def extract_mobile_number(text):
        import re
        match = re.search(r'\b\d{10}\b', text)  # Simplified pattern
        return match.group(0) if match else None

    @staticmethod
    def extract_skills(nlp_doc, noun_chunks):
        skills = []
        for token in nlp_doc:
            if token.pos_ == "NOUN":
                skills.append(token.text)
        return skills

    @staticmethod
    def get_number_of_pages(resume):
        if resume.endswith('.pdf'):
            reader = PdfReader(resume)
            return len(reader.pages)
        return 1  # Placeholder for non-PDF files


def resume_result_wrapper(resume):
    parser = ResumeParser(resume)
    return parser.get_extracted_data()


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())

    resumes = []
    for root, directories, filenames in os.walk('resumes'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    results = [pool.apply_async(resume_result_wrapper, args=(x,)) for x in resumes]
    results = [p.get() for p in results]

    pprint.pprint(results)
