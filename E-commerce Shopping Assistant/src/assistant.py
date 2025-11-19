import os
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 👇 Adjust if your CSV is elsewhere
DATA_PATH = r"C:\Users\hassa\Desktop\E-commerce-Shopping-Assistant\E-commerce Shopping Assistant\data\products.csv"
ORDERS_PATH = os.path.join(os.path.dirname(DATA_PATH), "orders.csv")


def find_column(df, candidate_names, required=False, default=None):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidate_names:
        cand = cand.lower()
        if cand in cols_lower:
            return cols_lower[cand]
    if required and default is None:
        raise ValueError(f"Required column not found. Tried: {candidate_names}")
    return default


def load_and_prepare_products(path=DATA_PATH):
    df = pd.read_csv(path)

    id_col = find_column(df, ["product_id", "id", "sku", "productid"], required=False)
    name_col = find_column(df, ["name", "product_name", "title", "product_title"], required=True)
    category_col = find_column(df, ["category", "product_category", "department", "product_type"], required=False)
    price_col = find_column(df, ["price", "current_price", "sale_price", "retail_price", "actual_price"], required=True)
    desc_col = find_column(df, ["description", "product_description", "details", "detail",
                                "short_description", "long_description"], required=False)

    if id_col is not None:
        df.rename(columns={id_col: "product_id"}, inplace=True)
    else:
        df["product_id"] = range(1, len(df) + 1)

    # 👉 NEW: ensure product_id is numeric integer (fixes 126704571.0 issue)
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int64")

    df.rename(columns={name_col: "name", price_col: "price"}, inplace=True)

    if category_col is not None:
        df.rename(columns={category_col: "category"}, inplace=True)
    else:
        df["category"] = "Unknown"

    if desc_col is not None:
        df.rename(columns={desc_col: "description"}, inplace=True)
    else:
        df["description"] = ""

    df["name"] = df["name"].astype(str)
    df["category"] = df["category"].astype(str)
    df["description"] = df["description"].astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    df["text"] = df["name"] + " " + df["description"]

    return df


def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


def show_products(df):
    if df.empty:
        print("No products to show.")
        return

    cols = ["product_id", "name", "price"]
    if "category" in df.columns:
        cols.insert(2, "category")
    print(df[cols].to_string(index=False))


def search_products(query, products_df, vectorizer, tfidf_matrix, top_n=5):
    query = query.strip()
    if not query:
        print("Please type something to search.")
        return
    
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = scores.argsort()[::-1][:top_n]
    results = products_df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    
    print(f"Top {top_n} search results for: '{query}'")
    show_products(results)


def filter_by_category(category, products_df):
    category = category.strip().lower()
    if "category" not in products_df.columns:
        print("No 'category' column available in this dataset.")
        return

    mask = products_df["category"].str.lower() == category
    results = products_df[mask]
    
    if results.empty:
        print(f"No products found in category: {category}")
    else:
        print(f"Products in category: {category}")
        show_products(results)


def recommend_similar(product_id, products_df, similarity_matrix, top_n=3):
    if product_id not in products_df["product_id"].values:
        print("Invalid product_id. Please choose from the table.")
        return
    
    idx = products_df.index[products_df["product_id"] == product_id][0]
    scores = similarity_matrix[idx]
    
    similar_indices = scores.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    
    recs = products_df.iloc[similar_indices]
    print(f"Products similar to '{products_df.loc[idx, 'name']}':")
    show_products(recs)


# ------------ CART & CHECKOUT ------------

cart = []


def add_to_cart(product_id, quantity, products_df):
    if product_id not in products_df["product_id"].values:
        print("Invalid product_id.")
        return
    
    for item in cart:
        if item["product_id"] == product_id:
            item["quantity"] += quantity
            print(f"Updated quantity for product_id {product_id}.")
            return
    
    product = products_df[products_df["product_id"] == product_id].iloc[0]
    cart.append({
        "product_id": product_id,
        "name": product["name"],
        "price": product["price"],
        "quantity": quantity
    })
    print(f"Added to cart: {product['name']} (x{quantity})")


def view_cart():
    if not cart:
        print("Your cart is empty.")
        return
    
    df_cart = pd.DataFrame(cart)
    df_cart["total"] = df_cart["price"] * df_cart["quantity"]
    print("\n🛒 Your cart:")
    print(df_cart.to_string(index=False))
    print(f"\nGrand total: {df_cart['total'].sum()}")


def clear_cart():
    cart.clear()
    print("Cart cleared.")


def checkout(save_path=ORDERS_PATH):
    """Save cart to orders.csv and clear it."""
    if not cart:
        print("Your cart is empty. Add items before checkout.")
        return
    
    df_cart = pd.DataFrame(cart)
    df_cart["total"] = df_cart["price"] * df_cart["quantity"]
    df_cart["order_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    df_cart["order_timestamp"] = datetime.now().isoformat(timespec="seconds")

    cols = ["order_id", "order_timestamp", "product_id", "name", "price", "quantity", "total"]
    df_cart = df_cart[cols]

    if os.path.exists(save_path):
        df_cart.to_csv(save_path, mode="a", header=False, index=False)
    else:
        df_cart.to_csv(save_path, index=False)

    print(f"\n✅ Order saved to {save_path}")
    print(f"Order ID: {df_cart['order_id'].iloc[0]}")
    clear_cart()


# ------------ SALES ANALYSIS (CLI) ------------

def load_orders(path=ORDERS_PATH):
    """
    Load orders.csv if it exists, else show a message.
    """
    if not os.path.exists(path):
        print(f"\nNo orders file found at: {path}")
        print("👉 Make at least one checkout first to create orders.csv.")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print("\norders.csv exists but has no rows yet.")
        return None

    if "order_timestamp" in df.columns:
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_date"] = df["order_timestamp"].dt.date

    return df


def show_basic_kpis_cli(orders_df):
    if orders_df is None or orders_df.empty:
        print("\nNo order data available for KPIs.")
        return
    
    total_revenue = orders_df["total"].sum()
    total_order_lines = len(orders_df)
    unique_orders = orders_df["order_id"].nunique()
    order_revenue = orders_df.groupby("order_id")["total"].sum()
    aov = order_revenue.mean()
    unique_products_sold = orders_df["product_id"].nunique()

    print("\n===== KEY METRICS (KPIs) =====")
    print(f"💰 Total Revenue:         {total_revenue}")
    print(f"🧾 Total Order Lines:     {total_order_lines}")
    print(f"📦 Unique Orders:         {unique_orders}")
    print(f"💳 Avg Order Value (AOV): {round(aov, 2)}")
    print(f"🛍️ Unique Products Sold:  {unique_products_sold}")


def top_products_by_revenue_cli(orders_df, top_n=10):
    if orders_df is None or orders_df.empty:
        print("\nNo order data available for product analysis.")
        return
    
    product_revenue = (
        orders_df
        .groupby(["product_id", "name"], as_index=False)
        .agg(
            total_revenue=("total", "sum"),
            total_quantity=("quantity", "sum")
        )
        .sort_values("total_revenue", ascending=False)
    )

    print(f"\n===== Top {top_n} Products by Revenue =====")
    print(product_revenue.head(top_n).to_string(index=False))


def revenue_by_category_cli(orders_df, products_df):
    if orders_df is None or orders_df.empty:
        print("\nNo order data available for category analysis.")
        return

    # Copy to avoid modifying originals
    orders_df = orders_df.copy()
    products_df = products_df.copy()

    orders_df["product_id"] = orders_df["product_id"].astype(str)
    products_df["product_id"] = products_df["product_id"].astype(str)

    cat_col = find_column(products_df, ["category", "product_category", "department", "product_type"])
    if cat_col is None:
        print("\nNo category column found in products.csv, skipping category analysis.")
        return

    products_df.rename(columns={cat_col: "category"}, inplace=True)

    merged = orders_df.merge(
        products_df[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    cat_rev = (
        merged
        .groupby("category", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
    )

    print("\n===== Revenue by Category =====")
    print(cat_rev.to_string(index=False))


def run_sales_analysis_cli(products_df):
    """
    Run the full sales analysis in CLI.
    """
    print("\n📊 Running Sales Analysis...")
    print("Reading orders from:", ORDERS_PATH)
    orders_df = load_orders(ORDERS_PATH)

    if orders_df is None:
        return

    show_basic_kpis_cli(orders_df)
    top_products_by_revenue_cli(orders_df, top_n=10)
    revenue_by_category_cli(orders_df, products_df)


# ------------ MAIN CLI ASSISTANT ------------

def shopping_assistant():
    products_df = load_and_prepare_products(DATA_PATH)
    vectorizer, tfidf_matrix, similarity_matrix = build_model(products_df)

    print("👋 Welcome to the E-commerce Shopping Assistant (Python Script)!")
    print("Using CSV:", DATA_PATH)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. View sample of products")
        print("2. Search products")
        print("3. Filter by category")
        print("4. Recommend similar products")
        print("5. Add to cart")
        print("6. View cart")
        print("7. Clear cart")
        print("8. Checkout (save order)")
        print("9. Run Sales Analysis (orders.csv)")
        print("10. Exit")
        
        choice = input("Enter choice (1-10): ").strip()
        
        if choice == "1":
            show_products(products_df.head(20))
        
        elif choice == "2":
            query = input("Search for (e.g. 'shoes', 't-shirt', 'bag'): ")
            search_products(query, products_df, vectorizer, tfidf_matrix)
        
        elif choice == "3":
            cat = input("Enter category (exact name, e.g. 'Shoes', 'Clothing', 'Accessories'): ")
            filter_by_category(cat, products_df)
        
        elif choice == "4":
            try:
                pid = int(input("Enter product_id to get similar items: "))
                recommend_similar(pid, products_df, similarity_matrix)
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == "5":
            try:
                pid = int(input("Enter product_id to add to cart: "))
                qty = int(input("Enter quantity: "))
                add_to_cart(pid, qty, products_df)
            except ValueError:
                print("Please enter valid numbers for product_id and quantity.")
        
        elif choice == "6":
            view_cart()
        
        elif choice == "7":
            clear_cart()
        
        elif choice == "8":
            checkout()
        
        elif choice == "9":
            run_sales_analysis_cli(products_df)
        
        elif choice == "10":
            print("Thank you for shopping! 👋")
            break
        
        else:
            print("Invalid choice, please select 1–10.")


if __name__ == "__main__":
    shopping_assistant()
