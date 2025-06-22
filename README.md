# ProbNEL: Probabilistic NER‑Based Entity Linking

A flexible, transparent entity linking system that leverages Named Entity Recognition (NER) class probabilities, 
contextual embeddings, and DBpedia knowledge‑graph features to disambiguate and link mentions in text.

**Keywords:** entity linking, NER, NED, knowledge graphs, DBpedia, embeddings, Flask

---

## Table of Contents

1. [Features](#features)  
2. [Short Demo](#short-demo)
3. [How It Works](#how-it-works)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Running the Demo](#running-the-demo)  
5. [Usage](#usage)  
   - [Web GUI](#web-gui)  
   - [API](#api)  
6. [Candidate Selector trainig and selection](#candidate-selector-trainig-and-selection)
7. [License](#license)  
8. [Acknowledgements](#acknowledgements)  

---


## Short Demo

[![ProbNEL Demo](https://img.youtube.com/vi/mHKGdNv7XaM/0.jpg)](https://www.youtube.com/watch?v=mHKGdNv7XaM)

---

## Features

- **Multiple NER Models**: Choose from three NER models trained using [SpanMaker framework](https://github.com/tomaarsen/SpanMarkerNER):
  - [CoNLL++](https://huggingface.co/tomaarsen/span-marker-xlm-roberta-large-conll03-doc-context)
  - [OntoNotes 5.0](https://huggingface.co/tomaarsen/span-marker-roberta-large-ontonotes5)
  - [Few‑NERD](https://huggingface.co/tomaarsen/span-marker-bert-base-fewnerd-fine-super)  
- **Type‑Aware Disambiguation**: Optional embedding features based on predicted NER types.  
- **Feature‑Rich Ranking**: Combines string similarity, popularity, context embeddings, position, and type embeddings in an XGBoost model.  
- **Interactive GUI**:  
  - Highlighted, clickable entity mentions  
  - Accordion view of NER probabilities and candidate details  
  - Dynamic thumbnails from Wikipedia Commons   
- **Configurable**: select one of the available NER models and toggle using type‑score features during NED. 

---

## How It Works

1. **Input & Configuration**  
   - User enters text.
   - Selects NER model and whether to use type‑score features.  
2. **NER Stage**  
   - Text is sent via AJAX to the Flask backend.  
   - The chosen transformer model produces entity spans and class probabilities.  
3. **Candidate Retrieval**  
   - For each span, up to 10 candidates are fetched from the KB.  
4. **Feature Extraction**  
   - Compute Levenshtein, popularity, context similarity, position, and optional type‑embedding scores.  
5. **Ranking & Selection**  
   - Feature vector is scaled and passed through a pretrained XGBoost pipeline.  
   - Best candidate index is returned; others are ranked for inspection.  
6. **Interactive Display**  
   - Frontend highlights mentions, shows NER‑class badges, and an accordion of candidate cards with details.  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- `pip`  
- Virtual environment (recommended)

### Installation

```bash
git clone https://github.com/Danzigerrr/ProbNEL.git
cd ProbNEL
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
```

### Running the Demo

```bash
cd App/NEL_project
python flask_app.py
```

Open your browser at `http://127.0.0.1:5000/NEL_app`.

---

## Usage

### Web GUI

1. Paste text.
2. Select NER model and toggle “Use type‑score features.”
3. Click **Process text with DBpedia**.
4. View highlighted entities in text and expand accordions to inspect probabilities, ontology types, scores, and thumbnails.

### API

Send a `POST` to `/NEL_app` with form‑encoded parameters:

| Parameter         | Description               |
| ----------------- |---------------------------|
| `user_input`      | Raw text                  |
| `knowledge_graph` | `dbpedia`                 |
| `ner_model`       | Full NER model identifier |
| `use_types_score` | `0` or `1`                |

Response is JSON with `text`, `entities`, `probabilities`, and `candidates`.

---

## Candidate Selector trainig and selection

Candidate selector is an XGboost model which select the best candidate among the 10 candidates fetched from DBpedia for a recognized named entity in text.
The code used for trainig and evaluation of differnt configurations of candidate selector model is presented in [Candidate_selector.ipynb](./Jupyter_Notebooks/Candidate_selector.ipynb). 

In order to reuse the feature scores calcualted for each candidate in trainig and test datasets a zip file containig the calculted stores was created.
It is avaialbe for downloads using [the following link](https://drive.google.com/file/d/1ghbJGd47i735rAt7dE8jCuyQ_BsxJK_9/view?usp=sharing).

If you want to reruse these scores, download this zip file and place it insdie the `Jupyter_Notebooks` directory. Code for downloading and unzipping the zip file is already included in [Candidate_selector.ipynb](./Jupyter_Notebooks/Candidate_selector.ipynb).

---

## License

This project is licensed under the GNU GPL v3.0. See [LICENSE](LICENSE) for details.

