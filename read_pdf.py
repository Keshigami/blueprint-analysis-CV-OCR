import pypdfium2 as pdfium
import sys

def extract_text(pdf_path):
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        text = ""
        for i in range(len(pdf)):
            page = pdf[i]
            text += page.get_textpage().get_text_range() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = "/Users/keshigami/Caltech CTME/PGC AIML- ADL & Computer Vision/HeadstormAI_Job_Page/Headstorm - Computer Vision Engineer - PH version (1) - Copy (1).pdf"
    print(extract_text(pdf_path))
