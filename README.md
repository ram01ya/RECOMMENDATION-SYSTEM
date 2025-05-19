**Product Recommendation Assistant**

This is an interactive Product Recommendation Assistant web app built with Streamlit. It allows users to find the best matching products, similar alternatives, and related accessories based on text or voice input queries.

**Key Features**

webscrapping samsung and apple products from amazon

Supports both text and audio input for user queries using speech recognition.

Uses SentenceTransformer embeddings and FAISS for efficient semantic similarity search.

Extracts price constraints and model hints from queries to filter recommendations.

Recommends a main product, a similar product, and accessories based on the detected product category.

Displays product images, detailed descriptions, pricing, and provides Amazon purchase links.

Allows adding recommended items to a session-based shopping cart.

Implements synonym mapping for better category and product recognition.

**Dataset details**

product_id:
Unique identifier for each product.

category:
The category or type of the product (e.g., smartphone, laptop, headphone).

product_title:
The full title or name of the product as displayed to users.

brand:
The manufacturer or brand name of the product (e.g., Apple, Samsung).

model:
Specific model identifier or number of the product (e.g., iPhone 15, Galaxy S22).

features:
A text description listing the product’s key features and specifications.

price:
The retail price of the product in the respective currency (e.g., INR).

rating:
Average user rating or review score, usually on a scale (e.g., 1 to 5).

review_count:
Number of user reviews or ratings received by the product.

image_url:
URL link to an image of the product for display in the app.

main_name:
General name or common reference for the main product group or family.

Is_accessory:
Boolean or flag indicating whether the item is an accessory (e.g., charger, case) or a main product.

**Technologies Used**

Python, Streamlit (web UI framework)

FAISS (similarity search)

SentenceTransformers (embedding model)

SpeechRecognition (audio input)

pandas, nltk (data processing)
