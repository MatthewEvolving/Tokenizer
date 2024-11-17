import os
import sys
import tiktoken
from docx import Document
import PyPDF2
from transformers import pipeline
import torch

def read_text_file(file_path):
    """Reads a .txt file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def read_docx_file(file_path):
    """Reads a .docx file and returns its content."""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_pdf_file(file_path):
    """Reads a .pdf file and returns its content."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)
            else:
                print(f"Warning: No text found on page {page_num + 1}.")
    return '\n'.join(full_text)

def tokenize_text(text, encoding='cl100k_base'):
    """Tokenizes the input text using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    return tokens, enc

def detokenize_text(tokens, encoding='cl100k_base'):
    """Converts tokens back to string using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    text = enc.decode(tokens)
    return text

def summarize_text(text):
    """Summarizes the input text using Hugging Face's transformers."""
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1  # -1 means CPU
    summarizer = pipeline("summarization", device=device)
    # Hugging Face models have maximum input lengths; split text if necessary
    max_chunk = 1000  # Adjust based on the model's capacity
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

def get_word_count(text):
    """Returns the number of words in the text."""
    return len(text.split())

def get_char_count(text):
    """Returns the number of characters in the text."""
    return len(text)

def print_comparison_summary(original_text, decoded_text, tokens):
    """Prints a detailed comparison summary of the original and decoded text."""
    print("\n=== TOKENIZATION SUMMARY ===")
    print(f"Original Text Statistics:")
    print(f"- Word Count: {get_word_count(original_text)}")
    print(f"- Character Count: {get_char_count(original_text)}")
    
    print(f"\nDecoded Text Statistics:")
    print(f"- Word Count: {get_word_count(decoded_text)}")
    print(f"- Character Count: {get_char_count(decoded_text)}")
    
    print(f"\nToken Statistics:")
    print(f"- Total Tokens: {len(tokens)}")
    
    # Verify text matching
    texts_match = original_text == decoded_text
    print(f"\nText Verification:")
    print(f"- Exact Match: {'✓ YES' if texts_match else '✗ NO'}")
    
    if not texts_match:
        # Find where the texts start to differ
        min_len = min(len(original_text), len(decoded_text))
        for i in range(min_len):
            if original_text[i] != decoded_text[i]:
                print(f"- First difference at position {i}")
                print(f"  Original: '{original_text[i:i+50]}...'")
                print(f"  Decoded:  '{decoded_text[i:i+50]}...'")
                break
        
        # Compare lengths if different
        if len(original_text) != len(decoded_text):
            print(f"- Length mismatch: Original ({len(original_text)}) vs Decoded ({len(decoded_text)})")

def main():
    try:
        file_path = input('Enter the path to your file (e.g., c:\\temp\\file.txt): ').strip('"')

        if not os.path.isfile(file_path):
            print("The file does not exist. Please check the path and try again.")
            sys.exit(1)

        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        # Read the file based on its extension
        if file_extension == '.txt':
            text = read_text_file(file_path)
        elif file_extension in ['.docx', '.doc']:
            text = read_docx_file(file_path)
        elif file_extension == '.pdf':
            text = read_pdf_file(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            sys.exit(1)

        print("\n--- Original Text Preview ---")
        print(text[:1000] + ('...' if len(text) > 1000 else ''))

        # Tokenize the text
        tokens, encoder = tokenize_text(text)
        
        # Convert tokens back to string
        reconstructed_text = detokenize_text(tokens)
        print("\n--- Reconstructed Text Preview ---")
        print(reconstructed_text[:1000] + ('...' if len(reconstructed_text) > 1000 else ''))

        # Print detailed comparison summary
        print_comparison_summary(text, reconstructed_text, tokens)

        # Generate content summary if text is long enough
        if len(text.split()) > 100:  # Only summarize if there's enough content
            print("\nGenerating content summary...")
            try:
                summary = summarize_text(text)
                print("\n--- Content Summary ---")
                print(summary)
            except Exception as e:
                print(f"\nUnable to generate content summary: {str(e)}")
        else:
            print("\nText too short for meaningful summarization")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 