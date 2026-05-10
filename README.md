# Dashboard NLP - Agricultural Query Analysis

Dashboard Streamlit untuk menganalisis query terkait pertanian menggunakan NLP.

## 🚀 Cara Menjalankan Secara Lokal

1. **Clone Repository:**
   ```bash
   git clone https://github.com/rifqidzaki/Dasboard-NLP.git
   cd Dasboard-NLP
   ```

2. **Siapkan Data (PENTING):**
   Karena file dataset `query_agg.csv` berukuran ~1GB (melebihi batas GitHub), Anda perlu mendapatkannya secara manual:
   - Mintalah file `query_agg.csv` kepada pemilik repo.
   - Letakkan file `query_agg.csv` di folder utama (root) project ini.

3. **Install Dependensi:**
   Pastikan Anda sudah menginstall library yang diperlukan:
   ```bash
   pip install streamlit pandas scikit-learn pickle-mixin
   ```

4. **Jalankan Dashboard:**
   ```bash
   streamlit run dashboard_nlp.py
   ```

## 🛠️ Tech Stack
- **Python** (Logic)
- **Streamlit** (UI/Dashboard)
- **Scikit-learn** (NLP & ML Model)
- **Pandas** (Data Processing)
