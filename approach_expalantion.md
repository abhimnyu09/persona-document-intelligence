Our solution for the Persona-Driven Document Intelligence challenge is an efficient, offline-first NLP pipeline designed to extract highly relevant information from documents based on user context. It excels by focusing on semantic understanding rather than simple keyword matching, ensuring the output directly aligns with the user's role and task.

---
#### **Core Methodology: Semantic Search & Extractive Summarization**

Our approach is built on a foundation of sentence embeddings, which allow us to represent the meaning of text numerically. The process is broken into three key stages:

1.  **Structured Document Ingestion:** We begin by parsing the input PDFs using **`PyMuPDF`**, a high-performance library that extracts text while retaining structural metadata like page numbers. The text is then segmented into coherent chunks (paragraphs), forming a structured and searchable knowledge base. This ensures that every piece of information is tied back to its precise location.

2.  **Context-Aware Ranking:** The system's intelligence comes from its ability to understand user intent. We dynamically formulate a rich "intelligent query" by combining the input **`Persona`** and **`Job-to-be-Done`**. This query is then converted into a vector embedding using the lightweight **`all-MiniLM-L6-v2`** sentence-transformer model. We generate embeddings for every document chunk and calculate the **cosine similarity** between each chunk and the query. This score represents true semantic relevance. Document pages are then ranked based on the maximum relevance score of the chunks they contain, directly satisfying the "Section Relevance" scoring criterion.

3.  **Refined Text Generation:** To provide granular insights for the "Sub-section Analysis," we avoid slow, generative models. Instead, we use an efficient **extractive summarization** technique. For each top-ranked page, we identify the specific sentences that are most semantically similar to the original query. These key sentences are then concatenated to create a dense, highly relevant **`Refined Text`**. This method is not only fast but also ensures the output is factually grounded in the source document, directly addressing the "Sub-Section Relevance" criterion.

This methodology is fully self-contained within a Docker container, adhering to all CPU, memory, and offline constraints while delivering high-quality, prioritized insights.
