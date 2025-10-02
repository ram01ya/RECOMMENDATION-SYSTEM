import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize

# Ensure 'punkt' tokenizer is available
nltk.download('punkt', quiet=True)

# --- Caching functions ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")  # Dataset must be in the same folder
    df["text_for_faiss"] = df.apply(
        lambda row: f"{row['brand']} {row['model']} {row['product_title']} {row['features']} {row['category']}", axis=1
    )
    return df

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Optional: precompute embeddings locally and save as 'embeddings.npy' for faster load
@st.cache_data
def precompute_embeddings(df):
    embedder = load_embedder()
    embeddings = embedder.encode(df["text_for_faiss"].tolist(), convert_to_numpy=True)
    return embeddings

# --- Utility functions ---
def extract_price_constraint(query):
    query = query.lower()
    match = re.search(r"(under|below|less than)\s?(\d{3,6})", query)
    if match:
        return "under", int(match.group(2))
    match = re.search(r"(above|over|greater than|more than)\s?(\d{3,6})", query)
    if match:
        return "above", int(match.group(2))
    return None, None

def extract_model_hint(query):
    tokens = word_tokenize(query)
    return [t for t in tokens if re.match(r"[a-zA-Z]\d+[a-zA-Z]", t) or (t.isdigit() and len(t) <= 2)]

def generate_amazon_link(product_id, category):
    return f"https://www.amazon.in/dp/{product_id}" if category.lower() == 'smartphone' else f"https://www.amazon.com/dp/{product_id}"

def map_synonyms(text, synonyms_dict):
    text = text.lower()
    for key, values in synonyms_dict.items():
        for val in values:
            if val.lower() in text:
                return key
    return None

def extract_category(query, entity_synonyms):
    return map_synonyms(query, {k: v for k, v in entity_synonyms.items() if k not in ['apple', 'samsung']})

# --- Synonyms and accessory mapping ---
entity_synonyms = {
    'smartphone': ['smartphone', 'mobile', 'cellphone', 'phone', 'iphone', 'galaxy'],
    'laptop': ['laptop', 'notebook', 'macbook', 'ultrabook'],
    'laptop bag': ['laptop bag', 'laptop case'],
    'phone charger': ['phone charger'],
    'laptop charger': ['laptop charger'],
    'headphone': ['headphone', 'earbuds'],
    'case': ['case', 'phone case'],
    'mouse': ['mouse', 'wireless mouse'],
    'screen protector': ['screen protector', 'tempered glass'],
}

accessory_mapping = {
    "smartphone": ["phone charger", "headphone", "case", "screen protector"],
    "laptop": ["laptop charger", "headphone", "mouse", "laptop bag"]
}

# --- Cart initialization ---
if "cart" not in st.session_state:
    st.session_state.cart = []

def add_to_cart(item):
    st.session_state.cart.append(item)
    st.success(f"Added {item['brand']} {item['model']} to cart")

# --- Streamlit UI ---
st.title("🛍 Product Recommendation Assistant")

query = st.text_input("Enter your product query:")

if query:
    df = load_data()
    embedder = load_embedder()
    embeddings = precompute_embeddings(df)
    faiss_index = build_faiss_index(embeddings)
    id_to_row = dict(enumerate(df.index))

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = faiss_index.search(query_embedding, 30)
    retrieved_df = df.iloc[[id_to_row[i] for i in indices[0]]].copy()

    # Apply price filter
    cond, price_val = extract_price_constraint(query)
    if cond == "under":
        retrieved_df = retrieved_df[retrieved_df["price"] <= price_val]
    elif cond == "above":
        retrieved_df = retrieved_df[retrieved_df["price"] >= price_val]

    # Filter by model hints
    model_hints = extract_model_hint(query)
    if model_hints:
        hint = model_hints[0].lower()
        main_df = retrieved_df[retrieved_df["model"].str.lower() == hint]
        if main_df.empty:
            main_df = retrieved_df[retrieved_df["model"].str.lower().str.startswith(hint)]
    else:
        main_df = pd.DataFrame()

    if main_df.empty:
        main_df = retrieved_df.head(1)

    if not main_df.empty:
        main_row = main_df.iloc[0]
        st.subheader("✅ Main Product")
        st.image(main_row["image_url"], width=150)
        st.markdown(f"**{main_row['brand']} {main_row['model']}** - ₹{main_row['price']}")
        st.caption(main_row["product_title"])
        st.write(main_row["features"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"[Buy Now]({generate_amazon_link(main_row['product_id'], main_row['category'])})")
        with col2:
            if st.button("Add to Cart", key="main_product"):
                add_to_cart(main_row)

        # Similar product
        detected_category = extract_category(query, entity_synonyms)
        top_model = main_row["model"].lower()
        top_embedding = embedder.encode([main_row["features"]], convert_to_numpy=True)
        similar_df = df[(df["category"].str.lower() == detected_category) & (df["model"].str.lower() != top_model)].copy()

        if not similar_df.empty:
            sim_embeddings = embedder.encode(similar_df["text_for_faiss"].tolist(), convert_to_numpy=True)
            sim_index = faiss.IndexFlatL2(embeddings.shape[1])
            sim_index.add(sim_embeddings)
            _, sim_indices = sim_index.search(top_embedding, 1)
            similar_row = similar_df.iloc[sim_indices[0][0]]

            st.subheader("🔁 Similar Product")
            st.image(similar_row["image_url"], width=150)
            st.markdown(f"**{similar_row['brand']} {similar_row['model']}** - ₹{similar_row['price']}")
            st.caption(similar_row["product_title"])
            st.write(similar_row["features"])
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"[Buy Now]({generate_amazon_link(similar_row['product_id'], similar_row['category'])})")
            with col4:
                if st.button("Add to Cart", key="similar_product"):
                    add_to_cart(similar_row)

        # Accessory Recommendations
        if detected_category in accessory_mapping:
            st.subheader("🎒 Accessories")
            seen_accessories = set()

            for acc in accessory_mapping[detected_category]:
                acc_query = f"{main_row['brand']} {top_model} {acc}"
                acc_embedding = embedder.encode([acc_query], convert_to_numpy=True)
                _, acc_indices = faiss_index.search(acc_embedding, 15)
                acc_df = df.iloc[[id_to_row[i] for i in acc_indices[0]]].copy()
                acc_df = acc_df[acc_df["category"].str.lower().str.contains(acc.lower())]
                acc_df = acc_df[~acc_df["product_id"].isin(seen_accessories)]
                if acc_df.empty:
                    continue

                acc_row = acc_df.iloc[0]
                seen_accessories.add(acc_row["product_id"])

                with st.expander(f"🧩 {acc.title()}"):
                    st.image(acc_row["image_url"], width=150)
                    st.markdown(f"**{acc_row['brand']} {acc_row['model']}** - ₹{acc_row['price']}")
                    st.caption(acc_row["product_title"])
                    st.write(acc_row["features"])
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown(f"[Buy Now]({generate_amazon_link(acc_row['product_id'], acc_row['category'])})")
                    with col6:
                        if st.button(f"Add to Cart - {acc}", key=f"{acc}_cart"):
                            add_to_cart(acc_row)

# Cart sidebar
st.sidebar.header("🛒 Your Cart")
if st.session_state.cart:
    for item in st.session_state.cart:
        st.sidebar.markdown(f"- {item['brand']} {item['model']} - ₹{item['price']}")
else:
    st.sidebar.info("Cart is empty.")
