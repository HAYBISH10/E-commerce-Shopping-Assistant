import os
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "products.csv")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
USERS_PATH = os.path.join(DATA_DIR, "users.csv")


# -------------------------
# UTIL
# -------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def hash_password(password: str) -> str:
    """Return SHA256 hash of a password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# -------------------------
# AUTH (USERS CSV)
# -------------------------
def init_auth():
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None


def load_users():
    """
    Load users from CSV. If file doesn't exist, create it with a default admin.
    users.csv columns: username, password_hash, role
    """
    ensure_data_dir()

    if not os.path.exists(USERS_PATH):
        # Create default admin
        df = pd.DataFrame(
            [
                {
                    "username": "Haybish",
                    "password_hash": hash_password("Haybish@172"),
                    "role": "admin",
                }
            ]
        )
        df.to_csv(USERS_PATH, index=False)
        return {"admin": {"password_hash": df.iloc[0]["password_hash"], "role": "admin"}}

    df = pd.read_csv(USERS_PATH)
    users = {}
    for _, row in df.iterrows():
        users[row["username"]] = {
            "password_hash": row["password_hash"],
            "role": row["role"],
        }
    return users


def save_users_dict(users: dict):
    """
    Save users dict back to CSV.
    users = {username: {password_hash:..., role:...}}
    """
    ensure_data_dir()
    rows = []
    for username, data in users.items():
        rows.append(
            {
                "username": username,
                "password_hash": data["password_hash"],
                "role": data["role"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(USERS_PATH, index=False)


def register_user(username: str, password: str):
    """
    Register a new user as 'customer'.
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    users = load_users()

    if username in users:
        return False, "Username already exists. Choose another."

    users[username] = {
        "password_hash": hash_password(password),
        "role": "customer",
    }
    save_users_dict(users)
    return True, "Registration successful! You can now log in."


def login(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if not user:
        return False, "User not found."

    if hash_password(password) != user["password_hash"]:
        return False, "Incorrect password."

    st.session_state.auth_user = username
    st.session_state.auth_role = user["role"]
    return True, f"Logged in as {username} ({user['role']})"


def logout():
    st.session_state.auth_user = None
    st.session_state.auth_role = None


# -------------------------
# DATA LOADING & PREP
# -------------------------
def find_column(df, candidate_names, required=False, default=None):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidate_names:
        cand = cand.lower()
        if cand in cols_lower:
            return cols_lower[cand]
    if required and default is None:
        raise ValueError(f"Required column not found. Tried: {candidate_names}")
    return default


@st.cache_data
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

    # Make product_id integer-like (fix 126704571.0 issue)
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


@st.cache_resource
def build_model(products_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(products_df["text"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


def load_orders(path=ORDERS_PATH):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    if "order_timestamp" in df.columns:
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_date"] = df["order_timestamp"].dt.date

    return df


# -------------------------
# CART HELPERS (SESSION)
# -------------------------
def init_session():
    if "cart" not in st.session_state:
        st.session_state.cart = []


def add_to_cart(product, quantity: int):
    if quantity <= 0:
        st.warning("Quantity must be at least 1.")
        return

    for item in st.session_state.cart:
        if item["product_id"] == product["product_id"]:
            item["quantity"] += quantity
            st.success(f"Updated quantity for {product['name']}")
            return

    st.session_state.cart.append({
        "product_id": product["product_id"],
        "name": product["name"],
        "price": product["price"],
        "quantity": quantity,
    })
    st.success(f"Added to cart: {product['name']} (x{quantity})")


def cart_to_df():
    if not st.session_state.cart:
        return pd.DataFrame(columns=["product_id", "name", "price", "quantity", "total"])
    df_cart = pd.DataFrame(st.session_state.cart)
    df_cart["total"] = df_cart["price"] * df_cart["quantity"]
    return df_cart


def checkout():
    df_cart = cart_to_df()
    if df_cart.empty:
        st.warning("Cart is empty. Add items before checkout.")
        return

    df_cart["order_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    df_cart["order_timestamp"] = datetime.now().isoformat(timespec="seconds")

    cols = ["order_id", "order_timestamp", "product_id", "name", "price", "quantity", "total"]
    df_cart = df_cart[cols]

    ensure_data_dir()
    if os.path.exists(ORDERS_PATH):
        df_cart.to_csv(ORDERS_PATH, mode="a", header=False, index=False)
    else:
        df_cart.to_csv(ORDERS_PATH, index=False)

    st.success(f"Order saved! Order ID: {df_cart['order_id'].iloc[0]}")
    st.session_state.cart = []


# -------------------------
# SALES ANALYSIS HELPERS
# -------------------------
def compute_kpis(orders_df):
    total_revenue = orders_df["total"].sum()
    total_order_lines = len(orders_df)
    unique_orders = orders_df["order_id"].nunique()
    order_revenue = orders_df.groupby("order_id")["total"].sum()
    aov = order_revenue.mean()
    unique_products_sold = orders_df["product_id"].nunique()

    return {
        "total_revenue": total_revenue,
        "total_order_lines": total_order_lines,
        "unique_orders": unique_orders,
        "aov": aov,
        "unique_products_sold": unique_products_sold,
    }


# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="E-commerce Shopping Assistant", layout="wide")
    ensure_data_dir()
    init_auth()
    init_session()

    # -------- SIDEBAR AUTH --------
    with st.sidebar:
        st.title("🔐 Account")

        if st.session_state.auth_user:
            st.success(f"Logged in as {st.session_state.auth_user} ({st.session_state.auth_role})")
            if st.button("Logout"):
                logout()
                st.rerun()
        else:
            mode = st.radio("Choose:", ["Login", "Register"])

            if mode == "Login":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.button("Login"):
                    ok, msg = login(username, password)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            elif mode == "Register":
                new_username = st.text_input("New username")
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                if st.button("Register"):
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(new_username, new_password)
                        if ok:
                            st.success(msg)
                            # Auto-login after registration
                            st.session_state.auth_user = new_username
                            st.session_state.auth_role = "customer"
                            st.rerun()
                        else:
                            st.error(msg)

        st.markdown("---")
        st.caption("Default admin: `admin / admin123`")

    # If not logged in, show simple landing page
    if not st.session_state.auth_user:
        st.title("🛒 E-commerce Shopping Assistant")
        st.subheader("Please login or register from the sidebar to continue.")
        st.stop()

    # -------- MAIN APP CONTENT --------
    st.title("🛒 E-commerce Shopping Assistant")
    st.caption("Search • Filter • Recommend • Cart • Checkout • Analyze")

    # Load data & model
    try:
        products_df = load_and_prepare_products(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading products.csv: {e}")
        st.stop()

    vectorizer, tfidf_matrix, similarity_matrix = build_model(products_df)

    # Sidebar filters (only for logged-in users)
    st.sidebar.header("Filters & Options")

    search_query = st.sidebar.text_input("Search products", placeholder="e.g. shoes, coat, Nike...")
    all_categories = ["All"] + sorted(products_df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by category", all_categories)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Cart")
    if st.sidebar.button("Clear cart"):
        st.session_state.cart = []

    # Tabs: admin sees Analytics, customer sees only first 2
    if st.session_state.auth_role == "admin":
        tab1, tab2, tab3 = st.tabs(["🧾 Browse & Search", "🛒 Cart & Checkout", "📊 Sales Analytics"])
    else:
        tab1, tab2 = st.tabs(["🧾 Browse & Search", "🛒 Cart & Checkout"])
        tab3 = None

    # ------------- TAB 1: Browse & Search -------------
    with tab1:
        st.subheader("Browse Products")

        df_view = products_df.copy()

        # Category filter
        if selected_category != "All":
            df_view = df_view[df_view["category"] == selected_category]

        # Search filter (TF-IDF)
        if search_query.strip():
            query_vec = vectorizer.transform([search_query.strip()])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            df_view = df_view.copy()
            df_view["score"] = scores
            df_view = df_view.sort_values("score", ascending=False)

        st.dataframe(
            df_view[["product_id", "name", "category", "price"]].head(50),
            use_container_width=True,
        )

        st.markdown("### ➕ Add to Cart")

        col_pid, col_qty, col_btn = st.columns([2, 1, 1])
        with col_pid:
            pid_input = st.text_input("Product ID", placeholder="Type product_id from the table above")
        with col_qty:
            qty_input = st.number_input("Quantity", min_value=1, max_value=100, value=1, step=1)
        with col_btn:
            if st.button("Add"):
                if not pid_input.strip():
                    st.warning("Please enter a product_id first.")
                else:
                    try:
                        pid_val = int(pid_input)
                        if pid_val not in products_df["product_id"].values:
                            st.error("Invalid product_id. Check the table above.")
                        else:
                            product_row = products_df[products_df["product_id"] == pid_val].iloc[0]
                            add_to_cart(product_row, qty_input)
                    except ValueError:
                        st.error("Product ID must be an integer.")

        st.markdown("### 🤖 Recommendations")
        rec_pid_options = products_df["product_id"].dropna().unique().tolist()
        rec_pid = st.selectbox("Select a product_id for similar items", rec_pid_options)

        if st.button("Show Similar Products"):
            idx = products_df.index[products_df["product_id"] == rec_pid][0]
            scores = similarity_matrix[idx]
            similar_indices = scores.argsort()[::-1]
            similar_indices = [i for i in similar_indices if i != idx][:5]
            recs = products_df.iloc[similar_indices][["product_id", "name", "category", "price"]]
            st.write(f"Products similar to **{products_df.loc[idx, 'name']}**:")
            st.dataframe(recs, use_container_width=True)

    # ------------- TAB 2: Cart & Checkout -------------
    with tab2:
        st.subheader("🛒 Your Cart")
        df_cart = cart_to_df()

        if df_cart.empty:
            st.info("Cart is empty. Go to *Browse & Search* to add products.")
        else:
            st.dataframe(df_cart, use_container_width=True)
            st.markdown(f"### Grand Total: `{df_cart['total'].sum():.2f}`")

            if st.button("✅ Checkout & Save Order"):
                checkout()

    # ------------- TAB 3: Sales Analytics (admin only) -------------
    if tab3 is not None and st.session_state.auth_role == "admin":
        with tab3:
            st.subheader("📊 Sales Analytics (orders.csv)")

            orders_df = load_orders(ORDERS_PATH)
            if orders_df is None:
                st.info("No orders found yet. Make at least one checkout to see analytics.")
            else:
                kpis = compute_kpis(orders_df)

                col1, col2, col3 = st.columns(3)
                col4, col5 = st.columns(2)

                col1.metric("Total Revenue", f"{kpis['total_revenue']:.2f}")
                col2.metric("Unique Orders", kpis["unique_orders"])
                col3.metric("Total Order Lines", kpis["total_order_lines"])
                col4.metric("Avg Order Value (AOV)", f"{kpis['aov']:.2f}")
                col5.metric("Unique Products Sold", kpis["unique_products_sold"])

                st.markdown("### Revenue Over Time")
                if "order_date" in orders_df.columns:
                    daily_rev = (
                        orders_df.groupby("order_date")["total"]
                        .sum()
                        .reset_index()
                        .sort_values("order_date")
                    )
                    st.line_chart(daily_rev.set_index("order_date")["total"])
                else:
                    st.info("No order_date column available.")

                st.markdown("### Top Products by Revenue")
                prod_rev = (
                    orders_df.groupby(["product_id", "name"], as_index=False)
                    .agg(total_revenue=("total", "sum"))
                    .sort_values("total_revenue", ascending=False)
                    .head(10)
                )
                st.dataframe(prod_rev, use_container_width=True)

                st.markdown("### Revenue by Category")
                # join with products to get category
                orders_tmp = orders_df.copy()
                prods_tmp = products_df.copy()
                orders_tmp["product_id"] = orders_tmp["product_id"].astype(str)
                prods_tmp["product_id"] = prods_tmp["product_id"].astype(str)

                cat_col = find_column(prods_tmp, ["category", "product_category", "department", "product_type"])
                if cat_col is not None:
                    prods_tmp.rename(columns={cat_col: "category"}, inplace=True)
                    merged = orders_tmp.merge(
                        prods_tmp[["product_id", "category"]],
                        on="product_id",
                        how="left"
                    )
                    cat_rev = (
                        merged.groupby("category", as_index=False)["total"]
                        .sum()
                        .sort_values("total", ascending=False)
                    )
                    st.bar_chart(cat_rev.set_index("category")["total"])
                else:
                    st.info("No category column found in products.csv, cannot compute revenue by category.")


if __name__ == "__main__":
    main()
