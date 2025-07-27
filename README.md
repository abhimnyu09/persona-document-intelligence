# Persona-Driven Document Intelligence

This project is an intelligent document analysis system built for the "Connect What Matters" hackathon challenge. It extracts and prioritizes the most relevant sections from a collection of PDF documents based on a specific user `persona` and their `job-to-be-done`.

The system uses a lightweight, offline-first NLP pipeline powered by sentence-transformer models to perform semantic search, ensuring high relevance and fast, CPU-only performance.

## Features

* **Context-Aware Analysis:** Combines a user persona and job description into a rich query to find what truly matters.
* **Semantic Search:** Uses sentence embeddings to understand the meaning behind the text, not just keywords.
* **Automated Ranking:** Ranks document sections based on their relevance to the user's goal.
* **Extractive Summarization:** Pulls the most important sentences from top-ranked sections to provide a granular summary.
* **Offline & Secure:** Runs entirely within a Docker container with no internet access required after setup.
* **Efficient:** Optimized for CPU execution with a small memory footprint (<1GB).

## Technology Stack

* **Backend:** Python 3.9
* **Containerization:** Docker
* **NLP Model:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **PDF Processing:** `PyMuPDF`
* **Core Libraries:** `numpy`, `scikit-learn`

## Project Structure

To run this project, your directory must be set up as follows:

```
persona-document-intelligence/
├── input/
│   ├── pdfs/
│   │   └── (Place your PDF files here)
│   ├── persona.json
│   └── job.txt
├── output/
│   └── (Output files will be generated here)
├── main.py
├── requirements.txt
└── Dockerfile
```

## Setup and Execution

Follow these steps to set up and run the project.

### 1. Prerequisites

* You must have **Docker** installed and running on your system.

### 2. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/abhimnyu09/persona-document-intelligence.git
cd persona-document-intelligence
```

### 3. Prepare Input Files

Place your input files in the `input/` directory.

* **Documents:** Add 3-10 related PDF files to the `input/pdfs/` folder.

* **Persona:** Edit the `input/persona.json` file. It must define the user's role and focus areas. For example:

  ```json
  {
    "role": "Hardware Engineering Recruiter",
    "expertise": "Sourcing candidates for semiconductor and VLSI roles",
    "focus_areas": [
      "Verilog", "VHDL", "FPGA", "Circuit Design", "SPICE", "CMOS"
    ]
  }
  ```

* **Job-to-be-Done:** Edit the `input/job.txt` file. It must describe the concrete task. For example:

  ```txt
  Identify top candidates for our summer internship program in Digital and Analog circuit design. Prioritize students with relevant coursework and hands-on project experience using tools like Verilog or Cadence.
  ```

### 4. Build the Docker Image

Open a terminal in the project's root directory and run the following command. This will build the Docker image, download the NLP model, and install all dependencies. This step requires an internet connection.

```bash
docker build -t doc-analyst .
```

### 5. Run the Analysis

Execute the following command to run the analysis. This step runs completely offline. The command mounts your local `input` and `output` folders into the container and runs the script.

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-analyst \
  python main.py \
  --docs_dir /app/input/pdfs \
  --persona_file /app/input/persona.json \
  --job_file /app/input/job.txt \
  --output_file /app/output/result.json
```

***Note for Windows Users:** In Command Prompt or PowerShell, replace `$(pwd)` with `%cd%`.*

## Output

The result of the analysis will be saved as `result.json` inside the `output/` folder. The file will contain three main sections:

1. **`Metadata`**: Contains information about the input documents, the persona, the job, and the processing time.
2. **`Extracted Section`**: A ranked list of the most relevant pages ("sections") from the documents, sorted by importance.
3. **`Sub-section Analysis`**: A corresponding list providing a `Refined Text` summary for each of the top-ranked sections, created by extracting the most relevant sentences.
