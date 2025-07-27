import os
import json
import fitz  # PyMuPDF
import numpy as np
import argparse
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500  # Number of characters per chunk
TOP_SECTIONS = 5 # Number of top sections to extract
TOP_SENTENCES_FOR_REFINED_TEXT = 3 # Number of sentences for the refined summary

def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence System")
    parser.add_argument('--docs_dir', required=True, help="Directory containing PDF documents.")
    parser.add_argument('--persona_file', required=True, help="JSON file defining the persona.")
    parser.add_argument('--job_file', required=True, help="Text file describing the job-to-be-done.")
    parser.add_argument('--output_file', required=True, help="Path to save the output JSON result.")
    return parser

def load_inputs(persona_path, job_path):
    """Loads persona and job-to-be-done from files."""
    with open(persona_path, 'r') as f:
        persona = json.load(f)
    with open(job_path, 'r') as f:
        job_to_be_done = f.read().strip()
    return persona, job_to_be_done

def generate_intelligent_query(persona, job):
    """Combines persona and job into a single detailed query."""
    return f"As a {persona.get('role')} with expertise in {persona.get('expertise')}, I need to {job}"

def parse_and_chunk_documents(docs_dir):
    """Parses all PDFs in a directory and splits them into manageable chunks."""
    all_chunks = []
    for filename in os.listdir(docs_dir):
        if filename.lower().endswith('.pdf'):
            doc_path = os.path.join(docs_dir, filename)
            doc = fitz.open(doc_path)
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                if not text.strip():
                    continue
                
                # Simple chunking by character count
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk_text = text[i:i + CHUNK_SIZE]
                    all_chunks.append({
                        "text": chunk_text,
                        "doc_name": filename,
                        "page_num": page_num
                    })
    return all_chunks

def generate_refined_text(full_text, query_embedding, model):
    """Performs extractive summarization to get key sentences."""
    # Split the text into sentences
    sentences = [s.strip() for s in full_text.split('.') if s.strip()]
    if not sentences:
        return ""

    # Generate embeddings for sentences
    sentence_embeddings = model.encode(sentences)
    
    # Calculate similarity with the main query
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
    
    # Get top N most similar sentences
    top_indices = np.argsort(similarities)[-TOP_SENTENCES_FOR_REFINED_TEXT:]
    top_indices.sort() # Sort to maintain original order
    
    refined_text = " ".join([sentences[i] for i in top_indices])
    return refined_text + "." if refined_text else ""


def main():
    """Main execution function."""
    start_time = datetime.datetime.now()
    
    # 1. Parse Arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 2. Load Inputs and Generate Query
    persona, job = load_inputs(args.persona_file, args.job_file)
    intelligent_query = generate_intelligent_query(persona, job)
    print("✅ Inputs loaded and intelligent query generated.")

    # 3. Load Model
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model '{MODEL_NAME}' loaded.")
    
    # 4. Parse and Chunk Documents
    chunks = parse_and_chunk_documents(args.docs_dir)
    if not chunks:
        print("❌ No text could be extracted from the documents. Exiting.")
        return
    print(f"✅ Parsed {len(chunks)} text chunks from documents.")

    # 5. Generate Embeddings
    query_embedding = model.encode([intelligent_query])
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)
    print("✅ Generated embeddings for query and all chunks.")
    
    # 6. Calculate Relevance and Rank
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    for i, chunk in enumerate(chunks):
        chunk['score'] = similarities[i]
        
    # Group chunks by document and page to form "sections"
    sections = {}
    for chunk in chunks:
        key = (chunk['doc_name'], chunk['page_num'])
        if key not in sections:
            sections[key] = {'text': '', 'max_score': 0}
        sections[key]['text'] += chunk['text'] + " "
        sections[key]['max_score'] = max(sections[key]['max_score'], chunk['score'])
        
    # Sort sections by their highest chunk score
    sorted_sections = sorted(sections.items(), key=lambda item: item[1]['max_score'], reverse=True)
    
    # 7. Prepare Output
    output = {
        "Metadata": {
            "input_documents": os.listdir(args.docs_dir),
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": start_time.isoformat()
        },
        "Extracted Section": [],
        "Sub-section Analysis": []
    }

    print("✅ Generating final output...")
    for i, (key, data) in enumerate(sorted_sections[:TOP_SECTIONS], 1):
        doc_name, page_num = key
        
        # For 'Extracted Section'
        output["Extracted Section"].append({
            "document": doc_name,
            "page_number": page_num,
            "section_title": f"Content from page {page_num}", # Simplified section title
            "importance_rank": i
        })
        
        # For 'Sub-section Analysis'
        refined_text = generate_refined_text(data['text'], query_embedding, model)
        output["Sub-section Analysis"].append({
            "document": doc_name,
            "page_number": page_num,
            "refined_text": refined_text
        })

    # 8. Save Output
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=4)
        
    end_time = datetime.datetime.now()
    print(f"🎉 Success! Analysis complete. Results saved to {args.output_file}")
    print(f"⏱️ Total processing time: {end_time - start_time}")

if __name__ == "__main__":
    main()
