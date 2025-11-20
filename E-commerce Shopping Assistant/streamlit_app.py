import os
import hashlib
from datetime import datetime, date
from io import BytesIO

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
WISHLIST_PATH = os.path.join(DATA_DIR, "wishlist.csv")


# -------------------------
# UTIL
# -------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def save_products_df(df: pd.DataFrame):
    """Persist products DataFrame back to CSV and clear caches."""
    ensure_data_dir()
    df.to_csv(DATA_PATH, index=False)
    load_and_prepare_products.clear()
    build_model.clear()
    st.success("✅ Products saved & model cache refreshed. Please reload if needed.")


def orders_to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    """Convert orders dataframe to an in-memory Excel file."""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Orders")
    except ModuleNotFoundError:
        st.error("Excel export requires 'xlsxwriter'. Install it with: pip install xlsxwriter")
        return BytesIO()
    output.seek(0)
    return output


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
    Load users from CSV.

    If users.csv does NOT exist or is broken:
      → create it with a single admin:
           username: Hassani
           password: Hassani@172
           role: admin

    If users.csv exists but doesn't contain Hassani:
      → append Hassani as admin (keep other users).
    """
    ensure_data_dir()

    admin_username = "Hassani"
    admin_password_hash = hash_password("Hassani@172")

    # 1) If file is missing → create with only admin
    if not os.path.exists(USERS_PATH):
        df = pd.DataFrame(
            [
                {
                    "username": admin_username,
                    "password_hash": admin_password_hash,
                    "role": "admin",
                }
            ]
        )
        df.to_csv(USERS_PATH, index=False)
    else:
        df = pd.read_csv(USERS_PATH)

        # 2) If file is broken (missing columns) → reset to only admin
        required_cols = {"username", "password_hash", "role"}
        if not required_cols.issubset(df.columns):
            df = pd.DataFrame(
                [
                    {
                        "username": admin_username,
                        "password_hash": admin_password_hash,
                        "role": "admin",
                    }
                ]
            )
            df.to_csv(USERS_PATH, index=False)
        else:
            # 3) If Hassani is missing → append admin row (keep other users)
            usernames = df["username"].astype(str).tolist()
            if admin_username not in usernames:
                admin_row = pd.DataFrame(
                    [
                        {
                            "username": admin_username,
                            "password_hash": admin_password_hash,
                            "role": "admin",
                        }
                    ]
                )
                df = pd.concat([df, admin_row], ignore_index=True)
                df.to_csv(USERS_PATH, index=False)

    # 4) Reload final & build dict
    df = pd.read_csv(USERS_PATH)
    users = {}
    for _, row in df.iterrows():
        users[row["username"]] = {
            "password_hash": row["password_hash"],
            "role": str(row["role"]).lower(),
        }
    return users


def save_users_dict(users: dict):
    """
    Save users back to CSV.
    Admin can edit/delete users from the Admin Panel.
    """
    rows = []
    for username, data in users.items():
        rows.append(
            {
                "username": username,
                "password_hash": data["password_hash"],
                "role": str(data["role"]).lower(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(USERS_PATH, index=False)


def register_user(username: str, password: str):
    """
    Register new user directly as 'customer' (no admin approval needed).
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    # Prevent using main admin name from UI
    if username == "Hassani":
        return False, "Username 'Hassani' is reserved for admin."

    users = load_users()
    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password_hash": hash_password(password),
        "role": "customer",  # ✅ no more 'pending'
    }
    save_users_dict(users)
    return True, "Registration successful! You can now log in."


def login(username: str, password: str):
    """
    Generic login for BOTH user page and admin page.
    On the admin page we will additionally check role=='admin'.
    """
    users = load_users()
    user = users.get(username)
    if not user:
        return False, "User not found."

    if hash_password(password) != user["password_hash"]:
        return False, "Incorrect password."

    # ✅ no more blocking on 'pending' – everyone (non-admin) behaves like a normal user
    st.session_state.auth_user = username
    st.session_state.auth_role = user["role"]
    return True, f"Logged in as {username} ({user['role']})"


def logout():
    st.session_state.auth_user = None
    st.session_state.auth_role = None


# -------------------------
# DATA PROCESSING
# -------------------------
def find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


@st.cache_data
def load_and_prepare_products(path=DATA_PATH):
    df = pd.read_csv(path)

    id_col = find_column(df, ["product_id", "id", "sku", "productid"])
    name_col = find_column(df, ["name", "product_name", "title", "product_title"])
    category_col = find_column(df, ["category", "product_category", "department", "product_type"])
    price_col = find_column(df, ["price", "current_price", "sale_price", "retail_price", "actual_price"])
    desc_col = find_column(df, ["description", "product_description", "details", "detail",
                                "short_description", "long_description"])
    image_col = find_column(df, ["image_url", "image", "img", "picture", "image_link", "imageurl"])

    # product_id
    if id_col is None:
        df["product_id"] = range(1, len(df) + 1)
    else:
        df.rename(columns={id_col: "product_id"}, inplace=True)
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int64")

    # name + price
    df.rename(columns={name_col: "name", price_col: "price"}, inplace=True)

    # category
    if category_col:
        df.rename(columns={category_col: "category"}, inplace=True)
    else:
        df["category"] = "Unknown"

    # description
    if desc_col:
        df.rename(columns={desc_col: "description"}, inplace=True)
    else:
        df["description"] = ""

    # image_url
    if image_col:
        df.rename(columns={image_col: "image_url"}, inplace=True)
    else:
        df["image_url"] = np.nan

    df["name"] = df["name"].astype(str)
    df["category"] = df["category"].astype(str)
    df["description"] = df["description"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    df["text"] = df["name"] + " " + df["description"]

    # Remove duplicates
    df = df.drop_duplicates(subset=["name", "description"]).reset_index(drop=True)

    # Hard cap (avoid slow TF-IDF)
    max_products = 8000
    if len(df) > max_products:
        df = df.sample(max_products, random_state=42).reset_index(drop=True)

    return df


@st.cache_resource
def build_model(products_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(products_df["text"])
    return vectorizer, matrix


# -------------------------
# CART & WISHLIST
# -------------------------
def init_session():
    if "cart" not in st.session_state:
        st.session_state.cart = []


def add_to_cart(product, qty):
    if qty <= 0:
        st.warning("Quantity must be at least 1.")
        return

    for item in st.session_state.cart:
        if item["product_id"] == product["product_id"]:
            item["quantity"] += qty
            st.success("Updated quantity!")
            return

    st.session_state.cart.append({
        "product_id": product["product_id"],
        "name": product["name"],
        "price": product["price"],
        "quantity": qty,
    })
    st.success("Added to cart!")


def cart_to_df():
    if not st.session_state.cart:
        return pd.DataFrame(columns=["product_id", "name", "price", "quantity", "total"])
    df = pd.DataFrame(st.session_state.cart)
    df["total"] = df["price"] * df["quantity"]
    return df


def checkout():
    df_cart = cart_to_df()
    if df_cart.empty:
        st.warning("Cart is empty.")
        return

    df_cart["order_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    df_cart["order_timestamp"] = datetime.now().isoformat(timespec="seconds")
    df_cart["username"] = st.session_state.auth_user  # link orders to customer

    df_cart.to_csv(ORDERS_PATH, mode="a", index=False, header=not os.path.exists(ORDERS_PATH))

    st.session_state.cart = []
    st.success("Order saved!")


def load_wishlist(username: str) -> pd.DataFrame:
    """Load wishlist rows for a specific user."""
    if not os.path.exists(WISHLIST_PATH):
        return pd.DataFrame(columns=["username", "product_id", "name", "price", "added_at"])
    df = pd.read_csv(WISHLIST_PATH)
    if df.empty:
        return df
    return df[df["username"] == username].copy()


def add_to_wishlist(product, username: str):
    ensure_data_dir()
    if os.path.exists(WISHLIST_PATH):
        df = pd.read_csv(WISHLIST_PATH)
    else:
        df = pd.DataFrame(columns=["username", "product_id", "name", "price", "added_at"])

    if not df.empty:
        mask = (df["username"] == username) & (df["product_id"] == product["product_id"])
        if mask.any():
            st.info("This product is already in your wishlist.")
            return

    new_row = {
        "username": username,
        "product_id": product["product_id"],
        "name": product["name"],
        "price": product["price"],
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(WISHLIST_PATH, index=False)
    st.success("Added to wishlist! ❤️")


def remove_from_wishlist(product_id, username: str):
    """Remove a product from a user's wishlist."""
    if not os.path.exists(WISHLIST_PATH):
        st.warning("Wishlist is empty.")
        return

    df = pd.read_csv(WISHLIST_PATH)
    before = len(df)
    df = df[~((df["username"] == username) & (df["product_id"] == product_id))]
    after = len(df)
    df.to_csv(WISHLIST_PATH, index=False)

    if after < before:
        st.success("Removed from wishlist.")
    else:
        st.info("Item not found in wishlist.")


# -------------------------
# SALES ANALYSIS
# -------------------------
def load_orders():
    """
    Safely load orders.csv.

    - If file does not exist → return None.
    - If file is corrupted / not valid CSV → show a friendly error and return None.
    """
    if not os.path.exists(ORDERS_PATH):
        return None

    try:
        df = pd.read_csv(ORDERS_PATH)
    except pd.errors.ParserError:
        st.error(
            "⚠️ The orders file (orders.csv) is corrupted or not a valid CSV format.\n\n"
            "👉 Fix:\n"
            "1. Delete `orders.csv` from the `data` folder in your project, then\n"
            "2. Place a new order from the user side to recreate it."
        )
        return None

    if "order_timestamp" in df.columns:
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_date"] = df["order_timestamp"].dt.date
    return df


def compute_kpis(df):
    if df is None or df.empty:
        return {
            "total_revenue": 0,
            "total_lines": 0,
            "unique_orders": 0,
            "unique_products": 0,
            "aov": 0,
        }
    return {
        "total_revenue": df["total"].sum(),
        "total_lines": len(df),
        "unique_orders": df["order_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "aov": df.groupby("order_id")["total"].sum().mean(),
    }


# -------------------------
# USER PORTAL UI
# -------------------------
def render_user_portal(products_df, vectorizer, tfidf_matrix):
    """User page: login/register, browse, cart, my orders & wishlist."""
    st.title("🛍️ E-commerce Shopping Assistant – User Page")

    # Tabs for users only
    tab1, tab2, tab3 = st.tabs(["Browse", "Cart", "My Orders & Wishlist"])

    # -------- TAB 1: BROWSE --------
    with tab1:
        st.subheader("Browse Products")

        top_row = st.columns([2, 1, 1])
        with top_row[0]:
            search = st.text_input("🔍 Search products")
        with top_row[1]:
            category = st.selectbox("Category", ["All"] + sorted(products_df["category"].unique()))
        with top_row[2]:
            view_mode = st.radio("View mode", ["Cards", "Table"], horizontal=True)

        df_view = products_df.copy()

        if category != "All":
            df_view = df_view[df_view["category"] == category]

        if search.strip():
            qv = vectorizer.transform([search])
            scores = cosine_similarity(qv, tfidf_matrix).flatten()
            df_view = df_view.copy()
            df_view["score"] = scores
            df_view = df_view.sort_values("score", ascending=False)

        # TABLE VIEW
        if view_mode == "Table":
            st.dataframe(
                df_view[["product_id", "name", "category", "price"]].head(100),
                use_container_width=True,
            )

        # CARDS VIEW
        else:
            st.markdown("#### 🧩 Product Cards")
            max_cards = 12
            df_cards = df_view.head(max_cards)

            if df_cards.empty:
                st.info("No products to show.")
            else:
                n_cols = 3
                rows = list(df_cards.itertuples(index=False))
                for i in range(0, len(rows), n_cols):
                    cols = st.columns(n_cols)
                    for col, prod in zip(cols, rows[i:i + n_cols]):
                        with col:
                            st.markdown(f"**{prod.name}**")
                            st.caption(prod.category)
                            st.markdown(f"💲 **{prod.price}**")

                            # Image if available
                            img_url = getattr(prod, "image_url", None)
                            if img_url and str(img_url).strip() and str(img_url).lower() != "nan":
                                st.image(str(img_url), use_column_width=True)

                            # Small description
                            desc = getattr(prod, "description", "")
                            if desc and len(desc) > 80:
                                desc = desc[:80] + "..."
                            st.write(desc)

                            # Add to cart form
                            with st.form(f"add_cart_{prod.product_id}"):
                                qty = st.number_input(
                                    "Qty",
                                    min_value=1,
                                    max_value=20,
                                    value=1,
                                    key=f"qty_card_user_{prod.product_id}",
                                )
                                submitted_cart = st.form_submit_button("Add to cart")
                                if submitted_cart:
                                    product_row = products_df[products_df["product_id"] == prod.product_id].iloc[0]
                                    add_to_cart(product_row, qty)

                            # Add to wishlist button
                            if st.button("❤️ Add to wishlist", key=f"wish_user_{prod.product_id}"):
                                product_row = products_df[products_df["product_id"] == prod.product_id].iloc[0]
                                add_to_wishlist(product_row, st.session_state.auth_user)

        st.markdown("---")
        st.markdown("### 🤖 Recommendations")
        rec_pid_options = products_df["product_id"].dropna().unique().tolist()
        rec_pid = st.selectbox("Select product_id for similar items", rec_pid_options)

        if st.button("Show Similar Products", key="user_show_similar"):
            idx = products_df.index[products_df["product_id"] == rec_pid][0]
            product_vec = tfidf_matrix[idx]
            scores = cosine_similarity(product_vec, tfidf_matrix).flatten()
            similar_indices = scores.argsort()[::-1]
            similar_indices = [i for i in similar_indices if i != idx][:5]

            recs = products_df.iloc[similar_indices][["product_id", "name", "category", "price"]]
            st.write(f"Products similar to **{products_df.loc[idx, 'name']}**:")
            st.dataframe(recs, use_container_width=True)

    # -------- TAB 2: CART --------
    with tab2:
        st.subheader("🛒 Your Cart")
        df_cart = cart_to_df()
        st.dataframe(df_cart, use_container_width=True)

        if not df_cart.empty:
            st.write(f"### Total: {df_cart['total'].sum():.2f}")
            if st.button("Checkout"):
                checkout()

    # -------- TAB 3: MY ORDERS & WISHLIST --------
    with tab3:
        st.subheader("📦 My Orders")

        orders_df = load_orders()
        username = st.session_state.auth_user

        if orders_df is None or orders_df.empty or "username" not in orders_df.columns:
            st.info("You have no saved orders yet. Place an order to see history here.")
        else:
            user_orders = orders_df[orders_df["username"] == username].copy()
            if user_orders.empty:
                st.info("You have not placed any orders yet.")
            else:
                # Basic summary for this user
                k_user = compute_kpis(user_orders)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Spent", f"{k_user['total_revenue']:.2f}")
                c2.metric("Orders", k_user["unique_orders"])
                last_order_date = user_orders["order_timestamp"].max()
                c3.metric("Last Order", str(last_order_date))

                st.markdown("#### Your Order Lines")
                st.dataframe(
                    user_orders[["order_id", "order_timestamp", "product_id", "name", "price", "quantity", "total"]],
                    use_container_width=True,
                )

        st.markdown("---")
        st.subheader("❤️ My Wishlist")

        wishlist_df = load_wishlist(username)
        if wishlist_df.empty:
            st.info("Your wishlist is empty. Add items from the Browse tab.")
        else:
            for row in wishlist_df.itertuples(index=False):
                with st.container():
                    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
                    with c1:
                        st.write(f"**{row.name}**")
                        st.caption(f"Product ID: {row.product_id}")
                    with c2:
                        st.write(f"💲 {row.price}")
                    with c3:
                        if st.button("Add to cart", key=f"wish_add_user_{row.product_id}"):
                            # Find product in products_df
                            if row.product_id in products_df["product_id"].values:
                                prod = products_df[products_df["product_id"] == row.product_id].iloc[0]
                                add_to_cart(prod, 1)
                            else:
                                st.error("Product no longer exists in catalog.")
                    with c4:
                        if st.button("Remove", key=f"wish_rm_user_{row.product_id}"):
                            remove_from_wishlist(row.product_id, username)
                            st.rerun()   # ✅ updated


# -------------------------
# ADMIN PORTAL UI
# -------------------------
def render_admin_portal(products_df, vectorizer, tfidf_matrix):
    """Admin page: only visible when role == admin."""
    is_admin = (str(st.session_state.auth_role).lower() == "admin")

    st.title("👑 Admin Dashboard – E-commerce Shopping Assistant")

    if not st.session_state.auth_user:
        st.info("Please log in as admin from the sidebar.")
        return

    if not is_admin:
        st.error("You are logged in, but you are not an admin user.")
        return

    # Admin-only tabs
    tab4, tab5 = st.tabs(["Analytics", "Admin Panel"])

    # -------- TAB 4: ANALYTICS --------
    with tab4:
        st.subheader("📊 Sales Analytics")

        orders_df = load_orders()
        if orders_df is None or orders_df.empty:
            st.info("No orders yet.")
        else:
            # Join with products to get category
            orders_tmp = orders_df.copy()
            prods_tmp = products_df.copy()
            orders_tmp["product_id"] = orders_tmp["product_id"].astype(str)
            prods_tmp["product_id"] = prods_tmp["product_id"].astype(str)

            prods_tmp = prods_tmp[["product_id", "category"]].copy()
            merged = orders_tmp.merge(prods_tmp, on="product_id", how="left")

            # Date range filter
            if "order_date" in merged.columns:
                min_date = merged["order_date"].min()
                max_date = merged["order_date"].max()
                if not isinstance(min_date, date):
                    min_date = min_date.date()
                    max_date = max_date.date()
                st.markdown("#### Filters")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    date_range = st.date_input(
                        "Date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )
                    if isinstance(date_range, tuple):
                        start_date, end_date = date_range
                    else:
                        start_date, end_date = min_date, max_date
                # Category filter
                with col_f2:
                    all_cats = ["All"] + sorted(
                        merged["category"].dropna().astype(str).unique().tolist()
                    )
                    selected_cat = st.selectbox("Category filter", all_cats)

                df_filtered = merged.copy()
                df_filtered = df_filtered[
                    (df_filtered["order_date"] >= start_date)
                    & (df_filtered["order_date"] <= end_date)
                ]
                if selected_cat != "All":
                    df_filtered = df_filtered[df_filtered["category"] == selected_cat]
            else:
                df_filtered = merged.copy()
                st.info("No order_date column available, showing all data without date filter.")

            if df_filtered.empty:
                st.warning("No data for selected filters.")
            else:
                # KPIs on filtered data
                k = compute_kpis(df_filtered)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Revenue", f"{k['total_revenue']:.2f}")
                c2.metric("Unique Orders", k["unique_orders"])
                c3.metric("Products Sold", k["unique_products"])
                c4.metric("Avg Order Value", f"{k['aov']:.2f}")

                st.markdown("### Revenue Over Time")
                if "order_date" in df_filtered.columns:
                    daily = (
                        df_filtered.groupby("order_date")["total"]
                        .sum()
                        .reset_index()
                        .sort_values("order_date")
                    )
                    st.line_chart(daily.set_index("order_date")["total"])
                else:
                    st.info("No order_date available for chart.")

                # Export to Excel button (filtered data)
                st.markdown("### 📥 Export Orders")
                excel_bytes = orders_to_excel_bytes(df_filtered)
                st.download_button(
                    "Download filtered orders as Excel",
                    data=excel_bytes,
                    file_name="orders_filtered.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

    # -------- TAB 5: ADMIN PANEL --------
    with tab5:
        st.subheader("🧰 Admin Panel")

        subtab1, subtab2, subtab3, subtab4 = st.tabs(
            ["Manage Users", "User Activity", "Upload Products", "Edit Products"]
        )

        # --- Manage Users ---
        with subtab1:
            st.markdown("### 👥 Manage Users")
            users = load_users()
            users_df = pd.DataFrame(
                [{"username": u, "role": info["role"]} for u, info in users.items()]
            )
            st.dataframe(users_df, use_container_width=True)

            st.markdown("#### ✏️ Edit User")
            selected_user = st.selectbox("Select user", users_df["username"].tolist())
            # simplify roles to admin / customer for editing:
            new_role = st.selectbox("New role", ["admin", "customer"], index=1)

            new_pass = st.text_input("New password (leave blank to keep current)", type="password")

            col_u1, col_u3 = st.columns(2)
            with col_u1:
                if st.button("Update User"):
                    if selected_user not in users:
                        st.error("User not found in dict.")
                    else:
                        # Prevent demoting yourself
                        if (
                            selected_user == st.session_state.auth_user
                            and new_role.lower() != "admin"
                        ):
                            st.error("You cannot remove your own admin role.")
                        else:
                            users[selected_user]["role"] = new_role
                            if new_pass.strip():
                                users[selected_user]["password_hash"] = hash_password(
                                    new_pass.strip()
                                )
                            save_users_dict(users)
                            st.success("User updated.")
                            st.rerun()   # ✅ updated

            with col_u3:
                if st.button("Delete User"):
                    if selected_user == st.session_state.auth_user:
                        st.error("You cannot delete the currently logged-in admin.")
                    else:
                        users.pop(selected_user, None)
                        save_users_dict(users)
                        st.success("User deleted.")
                        st.rerun()   # ✅ updated

        # --- User Activity ---
        with subtab2:
            st.markdown("### 📊 User Activity (Orders & Wishlist)")

            users = load_users()
            all_usernames = list(users.keys())

            if not all_usernames:
                st.info("No users found.")
            else:
                selected_user = st.selectbox("Select user", all_usernames, key="activity_user")

                st.markdown(f"#### 👤 Activity for `{selected_user}`")

                # Orders for selected user
                orders_df = load_orders()
                if orders_df is None or orders_df.empty or "username" not in orders_df.columns:
                    st.info("No orders found.")
                else:
                    user_orders = orders_df[orders_df["username"] == selected_user].copy()
                    if user_orders.empty:
                        st.info("This user has no orders yet.")
                    else:
                        k_user = compute_kpis(user_orders)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Spent", f"{k_user['total_revenue']:.2f}")
                        c2.metric("Orders", k_user["unique_orders"])
                        last_order_date = user_orders["order_timestamp"].max()
                        c3.metric("Last Order", str(last_order_date))

                        st.markdown("##### Order Lines")
                        st.dataframe(
                            user_orders[
                                [
                                    "order_id",
                                    "order_timestamp",
                                    "product_id",
                                    "name",
                                    "price",
                                    "quantity",
                                    "total",
                                ]
                            ],
                            use_container_width=True,
                        )

                st.markdown("---")
                # Wishlist for selected user
                if os.path.exists(WISHLIST_PATH):
                    wish_all = pd.read_csv(WISHLIST_PATH)
                    user_wish = wish_all[wish_all["username"] == selected_user].copy()
                    st.markdown("##### Wishlist Items")
                    if user_wish.empty:
                        st.info("This user has no wishlist items.")
                    else:
                        st.dataframe(
                            user_wish[["product_id", "name", "price", "added_at"]],
                            use_container_width=True,
                        )
                else:
                    st.info("No wishlist data yet.")

        # --- Upload Products ---
        with subtab3:
            st.markdown("### 📤 Upload New products.csv")
            uploaded = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded is not None:
                try:
                    new_df = pd.read_csv(uploaded)
                    st.write("Preview of uploaded data:")
                    st.dataframe(new_df.head(), use_container_width=True)

                    if st.button("Replace products.csv with this file"):
                        new_df.to_csv(DATA_PATH, index=False)
                        load_and_prepare_products.clear()
                        build_model.clear()
                        st.success(
                            "products.csv replaced successfully. Reload the app to use new data."
                        )
                except Exception as e:
                    st.error(f"Error reading uploaded CSV: {e}")

        # --- Edit Products ---
        with subtab4:
            st.markdown("### ✏️ Edit Products")

            if products_df.empty:
                st.info("No products available.")
            else:
                edit_pid = st.selectbox(
                    "Select product_id to edit",
                    products_df["product_id"].dropna().unique().tolist(),
                )

                row_idx = products_df[products_df["product_id"] == edit_pid].index[0]
                row = products_df.loc[row_idx]

                name_edit = st.text_input("Name", value=row["name"])
                category_edit = st.text_input("Category", value=row["category"])
                price_edit = st.number_input(
                    "Price", min_value=0.0, value=float(row["price"]), step=0.01
                )
                desc_edit = st.text_area("Description", value=row["description"])
                image_edit = st.text_input(
                    "Image URL (optional)",
                    value=""
                    if pd.isna(row.get("image_url", ""))
                    else str(row.get("image_url", "")),
                )

                if st.button("Save Changes"):
                    products_df.at[row_idx, "name"] = name_edit
                    products_df.at[row_idx, "category"] = category_edit
                    products_df.at[row_idx, "price"] = price_edit
                    products_df.at[row_idx, "description"] = desc_edit
                    products_df.at[row_idx, "image_url"] = (
                        image_edit if image_edit.strip() else np.nan
                    )

                    save_products_df(products_df)
                    st.rerun()   # ✅ updated


# -------------------------
# MAIN
# -------------------------
def main():
    st.set_page_config(page_title="E-commerce Assistant", layout="wide")
    ensure_data_dir()
    init_auth()
    init_session()

    # ---------- Decide if we are in USER mode or ADMIN mode ----------
    # Normal URL:          http://localhost:8501           -> user page
    # Admin secret URL:    http://localhost:8501/?admin=1  -> admin page

    # ✅ Updated: use st.query_params instead of experimental_get_query_params
    qparams = st.query_params
    admin_mode = qparams.get("admin", ["0"])[0] == "1"

    # ---------- SIDEBAR AUTH ----------
    with st.sidebar:
        st.title("🔐 Account")

        if admin_mode:
            # --------------- ADMIN PAGE LOGIN ---------------
            if st.session_state.auth_user and str(st.session_state.auth_role).lower() == "admin":
                st.success(f"Admin logged in: {st.session_state.auth_user}")
                if st.button("Logout"):
                    logout()
                    st.rerun()
            else:
                st.info("Admin login only. Normal users cannot access this page.")
                u = st.text_input("Admin username")
                p = st.text_input("Admin password", type="password")
                if st.button("Admin Login"):
                    ok, msg = login(u, p)
                    if ok and str(st.session_state.auth_role).lower() == "admin":
                        st.success(msg)
                        st.rerun()
                    elif ok:
                        # Logged in but not admin → kick them out
                        logout()
                        st.error("You are not an admin.")
                    else:
                        st.error(msg)

            # st.caption("Default admin: `Hassani / Hassani@172`")

        else:
            # --------------- USER PAGE LOGIN / REGISTER ---------------
            if st.session_state.auth_user:
                st.success(f"Logged in as: {st.session_state.auth_user} ({st.session_state.auth_role})")
                if st.button("Logout"):
                    logout()
                    st.rerun()
            else:
                mode = st.radio("Choose action:", ["Login", "Register"])

                if mode == "Login":
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    if st.button("Login"):
                        ok, msg = login(u, p)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                if mode == "Register":
                    u = st.text_input("New Username")
                    p = st.text_input("Password", type="password")
                    c = st.text_input("Confirm Password", type="password")
                    if st.button("Register"):
                        if p != c:
                            st.error("Passwords do not match.")
                        else:
                            ok, msg = register_user(u, p)
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)

            st.caption("Note: admin has a separate hidden page.")

    # ---------- Load products + model (used by BOTH portals) ----------
    try:
        products_df = load_and_prepare_products(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading products.csv: {e}")
        return

    vectorizer, tfidf_matrix = build_model(products_df)

    # ---------- Render the correct portal ----------
    if admin_mode:
        # Admin page
        render_admin_portal(products_df, vectorizer, tfidf_matrix)
    else:
        # User page
        if not st.session_state.auth_user:
            st.title("🛒 E-commerce Shopping Assistant – User Page")
            st.info("Please login or register from the sidebar to continue.")
            return
        render_user_portal(products_df, vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()
