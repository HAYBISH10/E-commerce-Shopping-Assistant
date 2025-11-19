# 🛒 E-commerce Shopping Assistant  

A complete mini e-commerce intelligence project built with:

- 🐍 Python (CLI Assistant)
- 📓 Jupyter Notebook (experiments & analysis)
- 🌐 Streamlit Web App (UI with login, registration & analytics)
- 📊 CSV data (products + orders)
- 🤖 TF-IDF Search & Similarity (product recommendations)

---

## 📌 Features

### 🔍 Product Search (TF-IDF)
- Intelligent keyword-based product search  
- Uses `TfidfVectorizer` + cosine similarity  
- Works even with long product titles & descriptions  

### 🏷️ Category Filtering
- Filter products by category  
- Automatically detects category column from any Kaggle products CSV  

### 🤖 Smart Recommendations
- "Similar products" based on text embeddings  
- Finds top N most similar items to a chosen product  

### 🛒 Shopping Cart System (CLI + Streamlit)
- Add products to cart  
- Automatically updates quantities  
- View cart total  
- Clear cart  

### 💳 Checkout & Orders CSV
- Saves each checkout as order lines into `orders.csv`  
- Auto-adds:
  - `order_id`
  - `order_timestamp`
  - `total` = price × quantity  

### 📊 Sales Analytics (CLI + Streamlit)
- Total revenue  
- Unique orders  
- Total order lines  
- Average order value (AOV)  
- Unique products sold  
- Top products by revenue  
- Revenue by category  
- Daily revenue trend (Streamlit charts)

### 🌐 Streamlit Web App
- Login & Registration system:
  - Users register with username + password
  - Passwords stored hashed (`SHA256`) in `data/users.csv`
- Roles:
  - **admin** → can view analytics
  - **customer** → shopping only
- Tabs:
  - 🧾 Browse & Search  
  - 🛒 Cart & Checkout  
  - 📊 Sales Analytics (admin only)

---

## 🗂️ Project Structure

```text
E-commerce Shopping Assistant/
│
├── data/
│   ├── products.csv        # Your Kaggle / custom products dataset
│   ├── orders.csv          # Auto-created after first checkout
│   └── users.csv           # Auto-created when first user registers (includes default admin)
│
├── notebooks/
│   └── ecommerce_assistant.ipynb   # Jupyter version (assistant + analysis)
│
├── src/
│   └── assistant.py        # CLI assistant (terminal)
│
├── streamlit_app.py        # Streamlit web app (UI, login, analytics)
│
├── requirements.txt
└── README.md
