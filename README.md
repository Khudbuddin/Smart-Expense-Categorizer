# 🏦 Smart Expense Categorizer
> **An End-to-End Document Intelligence & Automated Financial Analytics Platform**  
> Ingests unstructured bank statements (PDFs and scans), extracts tabular transactions via Computer Vision and OCR, and classifies expenditures with 90% accuracy using a hybrid Rule + Calibrated Machine Learning pipeline.
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Template%20Matching-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR%20Engine-blue?style=flat)](https://github.com/tesseract-ocr/tesseract)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
---
## 📌 Problem & Motivation
Bank statements and ePassbooks in India are notoriously fragmented. Each bank (Canara, ICICI, SBI, Cooperative banks) enforces different margin geometries, non-standard transaction descriptions, multi-line wrapped narrative strings, and unpredictable table headers across pages. 
Traditional rule parsers break when margins shift, while generic LLMs are costly and leak sensitive financial data.
**Smart Expense Categorizer** solves this with a **deterministic, local-first pipeline**:
1. Automatically identifies the bank layout using **Computer Vision logo matching & IFSC OCR**.
2. Extracts tabular transaction data without hardcoded global coordinates using **page-aware region slicing**.
3. Classifies transactions into spending categories via a **3-tier hybrid engine (Regex rules $\rightarrow$ RapidFuzz string distance $\rightarrow$ Calibrated Random Forest ML)**.
4. Delivers financial analytics via a **secure, multi-user Streamlit dashboard**.
---
## 🏗️ Architecture & Processing Pipeline

Unstructured Statement (PDF / Scanned Image / CSV)
│
▼
┌───────────────────────────────────────────────────────────────┐
│ 1. Multi-Strategy Bank Identification (bank_detector.py)      │
│    • Strategy A: Top-region OCR → IFSC Regex Normalization   │
│    • Strategy B: OpenCV Template Matching (cv2.matchTemplate) │
│    • Strategy C: Layout keyword heuristic fallback            │
└───────────────────────────────┬───────────────────────────────┘
│ (Identified: e.g., CNRB, JSBL)
▼
┌───────────────────────────────────────────────────────────────┐
│ 2. Coordinate-Adaptive Document Extraction (bank_templates.py)│
│    • Page 1 offset: Skips customer profile block              │
│    • Continuation pages: Skips repeated headers               │
│    • Slices columns: Date, Description, Debit, Credit, Balance│
└───────────────────────────────┬───────────────────────────────┘
│ Clean tabular text
▼
┌───────────────────────────────────────────────────────────────┐
│ 3. Hybrid 3-Tier Classification Engine                        │
│    ├─ Tier 1: Vendor Regex Mapping (Swiggy, Zomato, IRCTC, etc)│
│    ├─ Tier 2: RapidFuzz Levenshtein string matching           │
│    └─ Tier 3: Calibrated Random Forest ML Model               │
│               (TF-IDF n-grams [1,3] + Scaled Amount feature)  │
└───────────────────────────────┬───────────────────────────────┘
│ 90% F1-score classified records
▼
┌───────────────────────────────────────────────────────────────┐
│ 4. Persistence & Interactive Analytics (Frontend / SQLite)     │
│    • SQLite CRUD storage (Users, Transactions, Metadata)      │
│    • Streamlit Dashboard (Spending charts, search, auth)      │
└───────────────────────────────────────────────────────────────┘

---
## ✨ Key Features & Technical Highlights
### 1. Computer Vision & Multi-Strategy Bank Detection
Instead of asking users to pick their bank, `backend/file_handler/bank_detector.py` uses a cascaded approach:
* **IFSC Region OCR:** Crops the top 35% of the document, normalizes common OCR character confusions (`O` $\rightarrow$ `0`), and extracts the 11-character Indian Financial System Code via `\b[A-Z]{4}0[A-Z0-9]{6}\b`.
* **OpenCV Template Matching:** If text is degraded, it executes normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`) against stored bank logos (`ICICI.png`, `PhonePe.png`) with a confidence threshold $\ge 0.70$.
### 2. Multi-Page Coordinate-Aware Parsing
Rather than guessing global positions, `backend/file_handler/bank_templates.py` utilizes custom bounding dictionaries:
* **Dynamic Y-offsets:** Applies different vertical thresholds for Page 1 (`page1_start_y: 1350` to bypass account holder metadata) versus continuation pages (`other_pages_start_y: 250`).
* **Column Slicing:** Extracts exact pixel/point boundaries for withdrawals, deposits, and running balances.
### 3. Multimodal Machine Learning Classifier
When vendor descriptions are noisy or unseen, the system routes records to `backend/Scripts/train_model.py`:
* **Combined Features (`ColumnTransformer`):**
  * `Description`: Sub-linear `TfidfVectorizer` capturing word and character n-grams $(1, 3)$ across $4,000$ features.
  * `Amount`: Normalized using `StandardScaler` to factor transaction magnitude into categorization.
* **Calibrated Random Forest (`CalibratedClassifierCV`):**
  * Uses 50 balanced decision trees calibrated via Sigmoid probability scaling, ensuring confidence score reliability.
  * Evaluated using 3-fold stratified cross-validation, achieving **90% accuracy** across major expense categories.
### 4. Interactive Streamlit Interface
* **Authentication:** SQLite-backed user signup, hashed authentication, and session persistence.
* **Visual Summaries:** Real-time spending distributions, category-wise breakdowns, monthly variance analytics, and tabular transaction search.
---
## 📁 Repository Structure
```text
Smart-Expense-Categorizer/
│
├── backend/
│   ├── Databases/
│   │   ├── database_manager.py     # SQLite transaction CRUD & user management
│   │   └── database_setup.py       # Table schema definitions
│   ├── file_handler/
│   │   ├── bank_detector.py        # OpenCV logo matching & IFSC OCR detection
│   │   ├── bank_templates.py       # Bounding coordinates for Canara, JSBL, etc.
│   │   ├── pdf_load.py             # pdfplumber / Tesseract extraction engine
│   │   ├── pdf_to_images.py        # Poppler PDF-to-image converter
│   │   ├── csv_load.py             # CSV fallback parser
│   │   └── logos/                  # Bank logo reference images for template matching
│   ├── rules/
│   │   └── category_rules.py       # Regex dictionary for Indian vendors & services
│   └── Scripts/
│       ├── preprocess.py           # Text cleaning, lemmatization, and regex sanitizer
│       ├── train_model.py          # TF-IDF + Calibrated Random Forest training script
│       ├── predicted.py            # Unified hybrid inference orchestrator
│       ├── model.pkl               # Serialized production pipeline
│       └── vectorizer.pkl          # Serialized TF-IDF feature extractor
│
├── Frontend/
│   ├── main.py                     # Streamlit entrypoint & page routing
│   ├── auth/                       # Login and user registration modules
│   └── pages/
│       ├── dashboard.py            # Financial overview & KPI metric cards
│       ├── graph.py                # Visual spending distributions (bar/pie charts)
│       └── history.py              # Paginated, filterable transaction log
│
├── assets/
│   └── styles.css                  # Custom UI theme styling
├── requirements.txt                # Production dependencies
└── README.md                       # Comprehensive documentation
⚙️ Setup & Installation
1. Prerequisites

  
Python 3.10 or higher

  
Tesseract OCR: Install from UB-Mannheim
 (Windows) or via apt-get install tesseract-ocr (Ubuntu/Debian).

  
Poppler: Required for pdf2image. Download and extract binaries, then ensure bin/ is added to your system PATH.

  
2. Clone the Repository
bash
git clone https://github.com/Khudbuddin/Smart-Expense-Categorizer.git
cd Smart-Expense-Categorizer
3. Create and Activate a Virtual Environment
bash
# Windows:
python -m venv venv
venv\Scripts\activate
# Linux / macOS:
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
bash
pip install -r requirements.txt
5. Initialize the Database
bash
python backend/Databases/database_setup.py
6. Run the Application
bash
streamlit run Frontend/main.py

Open your browser at http://localhost:8501.

📊 Supported Formats & Categories

  
Supported Formats: Digital PDF statements, scanned receipt images (PNG/JPG), and CSV exports.

  
Pre-configured Bank Templates: Canara Bank (CNRB), Janaseva Sahakari Bank (JSBL), ICICI, PhonePe transaction logs, and generic fallbacks.

  
Spending Categories:
  
  

  
🍔 Food & Dining (Swiggy, Zomato, local restaurants, supermarkets)

  
🚗 Travel & Transport (Uber, Ola, IRCTC, Rapido, fuel/petrol)

  
⚡ Utilities & Bills (Electricity, mobile recharges, broadband)

  
🛍️ Shopping & Retail (Amazon, Flipkart, clothing, electronics)

  
🏥 Healthcare & Medical (Pharmacies, diagnostic centers, hospitals)

  
💼 Investments & Transfers (UPI peer transfers, mutual funds, brokerage deposits)

  

  
🤝 Contributing & License

Contributions are welcome! If you'd like to add a coordinate template for a new bank or enhance the ML model, please open a PR or submit an issue.

Distributed under the MIT License. See LICENSE for more information.

---
### What to clean up in your repo when pushing this:
1. **Delete `poppler-25.12.0/` from the GitHub repo:** Adding operating system binaries bloats your repository size. Put `poppler*` in your `.gitignore` and document the installation step (already handled in the setup guide above).
2. **Delete `storage.db` from git:** Do not commit active SQLite database files containing personal records. Add `*.db` to `.gitignore`.
