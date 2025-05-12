# Consolidated Text-to-SQL Evaluation Script (E-commerce)
# Includes Two Similarity Scores: Quora-Roberta (Paraphrase) & NLI-DeBERTa (Entailment-like)

import pandas as pd
from sentence_transformers import CrossEncoder
import torch
import torch.nn.functional as F # For softmax if needed
import time
import json

# --- Placeholder for your imported function ---
# Assume this function exists and is imported, handling the LLM calls
# def abc_response(model_name: str, prompt: str) -> tuple[str, float, int, int]:
#     """
#     Calls the specified LLM model with the given prompt.
#     Returns: (response_text, execution_time, input_tokens, output_tokens)
#     """
#     # Replace with your actual implementation
#     pass

# --- Dummy function if you need to run the script structure without your import ---
# Comment this out if you have the real abc_response imported
def abc_response(model_name: str, prompt: str) -> tuple[str, float, int, int]:
    # print(f"--- Simulating Call to {model_name} ---") # Keep console clean
    simulated_response = f"Simulated response for {model_name}"
    # Basic simulation logic based on prompt content for e-commerce
    if "Generate SQL" in prompt:
        if "customers in london" in prompt.lower():
             simulated_response = "SELECT customer_id, name, email FROM customers WHERE city = 'London';"
        elif "average order value" in prompt.lower():
             simulated_response = "SELECT AVG(total_amount) AS average_order_value FROM orders WHERE order_status = 'Completed';"
        elif "top 5 products" in prompt.lower() and "sales" in prompt.lower():
            simulated_response = """SELECT p.name, SUM(oi.quantity * oi.price_per_unit) AS total_sales
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'Completed'
GROUP BY p.product_id, p.name
ORDER BY total_sales DESC
LIMIT 5;"""
        elif "orders" in prompt.lower() and "pending" in prompt.lower() and "last 7 days" in prompt.lower():
            simulated_response = "SELECT order_id, order_date, total_amount FROM orders WHERE order_status = 'Pending' AND order_date >= date('now', '-7 days');" # Assuming SQLite date functions
        elif "customers" in prompt.lower() and "widget x" in prompt.lower() and "widget y" in prompt.lower():
            simulated_response = """SELECT c.name
FROM customers c
JOIN orders o1 ON c.customer_id = o1.customer_id
JOIN order_items oi1 ON o1.order_id = oi1.order_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN orders o2 ON c.customer_id = o2.customer_id
JOIN order_items oi2 ON o2.order_id = oi2.order_id
JOIN products p2 ON oi2.product_id = p2.product_id
WHERE p1.name = 'Widget X' AND p2.name = 'Widget Y';"""
        elif "reviews" in prompt.lower() and "rating < 3" in prompt.lower():
             simulated_response = "SELECT r.review_text, p.name FROM reviews r JOIN products p ON r.product_id = p.product_id WHERE r.rating < 3;"
        elif "products" in prompt.lower() and "electronics" in prompt.lower() and "price between" in prompt.lower():
             simulated_response = "SELECT p.name, p.price FROM products p JOIN categories c ON p.category_id = c.category_id WHERE c.name = 'Electronics' AND p.price BETWEEN 500 AND 1000;"
        else:
            simulated_response = "SELECT product_id, name FROM products LIMIT 10;" # Fallback

    elif "Generate Question" in prompt:
        if "city = 'London'" in prompt:
            simulated_response = "List customers residing in London."
        elif "AVG(total_amount)" in prompt:
             simulated_response = "What is the average value of completed orders?"
        elif "ORDER BY total_sales DESC" in prompt:
             simulated_response = "Which 5 products generated the most sales revenue from completed orders?"
        elif "order_status = 'Pending'" in prompt and "date('now', '-7 days')" in prompt:
             simulated_response = "Show pending orders from the last week."
        elif "'Widget X'" in prompt and "'Widget Y'" in prompt:
            simulated_response = "Which customers purchased both 'Widget X' and 'Widget Y'?"
        elif "rating < 3" in prompt:
            simulated_response = "Show review text for products with ratings below 3 stars."
        elif "c.name = 'Electronics'" in prompt and "BETWEEN 500 AND 1000" in prompt:
             simulated_response = "List electronic products priced between 500 and 1000."
        else:
             simulated_response = "What are the first 10 products?" # Fallback
    else:
        simulated_response = "Unknown simulation case."

    # Cleanup potential markdown/fencing
    if simulated_response.startswith("```sql"):
        simulated_response = simulated_response.splitlines()[1:-1] # Remove fences
        simulated_response = "\n".join(simulated_response).strip()
    elif simulated_response.startswith("```"):
         simulated_response = simulated_response[3:-3].strip()

    simulated_time = round(0.5 + len(prompt.split()) / 500 + len(simulated_response.split()) / 100, 2)
    simulated_input_tokens = len(prompt.split())
    simulated_output_tokens = len(simulated_response.split())
    return simulated_response.strip(), simulated_time, simulated_input_tokens, simulated_output_tokens
# --- End Dummy Function ---


# --- Load Cross-Encoder Models ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

cross_encoder_qqp = None
cross_encoder_nli = None

try:
    # Model 1: Trained for paraphrase identification (Quora Question Pairs)
    cross_encoder_qqp = CrossEncoder('cross-encoder/quora-roberta-base', max_length=512, device=device)
    print("Cross-Encoder QQP model loaded successfully.")
except Exception as e:
    print(f"Error loading Cross-Encoder QQP model: {e}")

try:
    # Model 2: Trained for Natural Language Inference
    cross_encoder_nli = CrossEncoder('cross-encoder/nli-deberta-v3-base', max_length=512, device=device)
    print("Cross-Encoder NLI model loaded successfully.")
except Exception as e:
    print(f"Error loading Cross-Encoder NLI model: {e}")

# --- Define Model Identifiers ---
TEXT_TO_SQL_MODEL = 'your-text-to-sql-model-ecommerce-v1.0' # Replace
SQL_TO_TEXT_MODEL = 'your-sql-to-text-model-ecommerce-v1.0' # Replace


# --- Define E-commerce Database Schema ---
SCHEMA_DICT = {
    "customers": """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, -- Unique identifier for the customer
    name VARCHAR(100) NOT NULL, -- Customer's full name
    email VARCHAR(100) UNIQUE, -- Customer's email address
    phone VARCHAR(20), -- Customer's phone number
    address VARCHAR(255), -- Shipping address
    city VARCHAR(50), -- City of residence
    country VARCHAR(50), -- Country of residence
    join_date DATE -- Date the customer registered
);
""",
    "categories": """
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY, -- Unique identifier for the category
    name VARCHAR(100) NOT NULL UNIQUE, -- Name of the category (e.g., 'Electronics', 'Books', 'Clothing')
    description TEXT -- Optional description of the category
);
""",
    "products": """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, -- Unique identifier for the product
    name VARCHAR(150) NOT NULL, -- Name of the product
    description TEXT, -- Detailed description of the product
    price DECIMAL(10, 2) NOT NULL, -- Price of the product
    category_id INTEGER, -- Foreign key referencing categories table
    stock_quantity INTEGER DEFAULT 0, -- Current number of items in stock
    average_rating DECIMAL(3, 2), -- Average customer rating (denormalized, updated by triggers/batch)
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""",
    "orders": """
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, -- Unique identifier for the order
    customer_id INTEGER, -- Foreign key referencing customers table
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Date and time the order was placed
    order_status VARCHAR(20) NOT NULL DEFAULT 'Pending', -- Status (e.g., 'Pending', 'Processing', 'Shipped', 'Completed', 'Cancelled')
    total_amount DECIMAL(12, 2), -- Total amount for the order (calculated)
    shipping_address VARCHAR(255), -- Shipping address used for this specific order
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
""",
    "order_items": """
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY, -- Unique identifier for this line item
    order_id INTEGER, -- Foreign key referencing orders table
    product_id INTEGER, -- Foreign key referencing products table
    quantity INTEGER NOT NULL, -- Number of units of the product ordered
    price_per_unit DECIMAL(10, 2) NOT NULL, -- Price of the product at the time of order
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
""",
    "reviews": """
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY, -- Unique identifier for the review
    product_id INTEGER, -- Foreign key referencing products table
    customer_id INTEGER, -- Foreign key referencing customers table
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5), -- Customer rating (1-5 stars)
    review_text TEXT, -- Customer's written review
    review_date DATE DEFAULT CURRENT_DATE, -- Date the review was submitted
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""
}
FULL_SCHEMA_STRING = "-- E-commerce Database Schema --\n\n" + "\n\n".join(SCHEMA_DICT.values())


# --- Define Prompt Templates with E-commerce Few-Shot Examples ---

TEXT_TO_SQL_PROMPT_TEMPLATE = """Given the following E-commerce database schema:
{schema}

Generate a syntactically correct SQL query for the given user question. Follow the examples provided.

--- Examples ---

Example 1:
Relevant Schema:
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100), city VARCHAR(50));
User Question: "Find the names of customers who live in Paris."
SQL Query: SELECT name FROM customers WHERE city = 'Paris';

Example 2:
Relevant Schema:
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150), price DECIMAL(10, 2), category_id INTEGER);
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name VARCHAR(100));
User Question: "What are the names and prices of products in the 'Clothing' category?"
SQL Query: SELECT T1.name ,  T1.price FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T2.name  =  'Clothing';

Example 3:
Relevant Schema:
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total_amount DECIMAL(12, 2));
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100));
User Question: "Show the total order amount for each customer, listing the customer's name."
SQL Query: SELECT T2.name ,  SUM(T1.total_amount) FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T2.name;

Example 4:
Relevant Schema:
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date TIMESTAMP, order_status VARCHAR(20));
User Question: "How many orders were placed yesterday with status 'Shipped'?"
SQL Query: SELECT COUNT(*) FROM orders WHERE date(order_date) = date('now', '-1 day') AND order_status = 'Shipped';

--- End Examples ---

Now, generate the SQL query for this question:
User Question: "{user_question}"
SQL Query:"""


SQL_TO_TEXT_PROMPT_TEMPLATE = """Given the following E-commerce database schema:
{schema}

And the following SQL query:
SQL Query: "{sql_query}"

Generate the natural language question that this SQL query most likely answers. Be concise and clear, mirroring how a user might ask. Follow the examples provided.

--- Examples ---

Example 1:
Relevant Schema:
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150), price DECIMAL(10, 2));
SQL Query: "SELECT name FROM products WHERE price < 50.00;"
Generated Question: Which products cost less than $50?

Example 2:
Relevant Schema:
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_status VARCHAR(20));
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100));
SQL Query: "SELECT T2.name FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id WHERE T1.order_status  =  'Pending';"
Generated Question: List the names of customers who have pending orders.

Example 3:
Relevant Schema:
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150), average_rating DECIMAL(3, 2));
SQL Query: "SELECT name FROM products ORDER BY average_rating DESC LIMIT 3;"
Generated Question: What are the top 3 highest-rated products?

Example 4:
Relevant Schema:
CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150));
SQL Query: "SELECT T2.name ,  SUM(T1.quantity) FROM order_items AS T1 JOIN products AS T2 ON T1.product_id  =  T2.product_id GROUP BY T2.name HAVING SUM(T1.quantity) > 100;"
Generated Question: Which products have sold more than 100 units in total across all orders?

--- End Examples ---

Now, generate the question for this query:
SQL Query: "{sql_query}"
Generated Question:"""


# --- Define E-commerce Test Cases (Sample - Expand to 100+) ---
test_cases = [
    # Basic Lookups & Filters
    {"id": 1, "question": "Find customer details for email 'test@example.com'.", "reference_sql": "SELECT * FROM customers WHERE email = 'test@example.com';", "reference_q_from_sql": "Show all information for the customer with email 'test@example.com'."},
    {"id": 2, "question": "List products that cost more than $1000.", "reference_sql": "SELECT name, price FROM products WHERE price > 1000.00;", "reference_q_from_sql": "What are the names and prices of products exceeding $1000?"},
    {"id": 3, "question": "Show orders placed on May 1st, 2025.", "reference_sql": "SELECT order_id, customer_id, total_amount FROM orders WHERE date(order_date) = '2025-05-01';", "reference_q_from_sql": "Which orders were created on 2025-05-01?"},
    {"id": 4, "question": "Find reviews with a 5-star rating.", "reference_sql": "SELECT review_id, product_id, customer_id, review_text FROM reviews WHERE rating = 5;", "reference_q_from_sql": "Show the reviews that gave a 5-star rating."},

    # Joins
    {"id": 5, "question": "What products are in the 'Electronics' category?", "reference_sql": "SELECT T1.name FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T2.name  =  'Electronics';", "reference_q_from_sql": "List product names belonging to the 'Electronics' category."},
    {"id": 6, "question": "Show the order dates for orders placed by customer ID 42.", "reference_sql": "SELECT order_date FROM orders WHERE customer_id = 42;", "reference_q_from_sql": "What are the dates of orders made by customer 42?"},
    {"id": 7, "question": "List the products included in order ID 101.", "reference_sql": "SELECT T2.name, T1.quantity, T1.price_per_unit FROM order_items AS T1 JOIN products AS T2 ON T1.product_id = T2.product_id WHERE T1.order_id = 101;", "reference_q_from_sql": "What products (name, quantity, price) were part of order 101?"},
    {"id": 8, "question": "Find the names of customers who reviewed the product with ID 50.", "reference_sql": "SELECT T2.name FROM reviews AS T1 JOIN customers AS T2 ON T1.customer_id = T2.customer_id WHERE T1.product_id = 50;", "reference_q_from_sql": "Which customers wrote a review for product 50?"},

    # Aggregations
    {"id": 9, "question": "How many products are in the 'Books' category?", "reference_sql": "SELECT count(*) FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T2.name  =  'Books';", "reference_q_from_sql": "What is the total number of products classified under 'Books'?"},
    {"id": 10, "question": "What is the total value of all 'Completed' orders?", "reference_sql": "SELECT SUM(total_amount) FROM orders WHERE order_status = 'Completed';", "reference_q_from_sql": "Calculate the sum of total amounts for all completed orders."},
    {"id": 11, "question": "Find the average rating for product ID 75.", "reference_sql": "SELECT AVG(rating) FROM reviews WHERE product_id = 75;", "reference_q_from_sql": "What is the average customer rating given to product 75?"},
    {"id": 12, "question": "How many orders has each customer placed? Show customer name and count.", "reference_sql": "SELECT T2.name ,  count(T1.order_id) FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T2.name ORDER BY count(T1.order_id) DESC;", "reference_q_from_sql": "Count the number of orders per customer, showing their name."},

    # Complex / Edge Cases
    {"id": 13, "question": "List customers from 'USA' who haven't placed any orders.", "reference_sql": "SELECT name FROM customers WHERE country = 'USA' AND customer_id NOT IN (SELECT DISTINCT customer_id FROM orders WHERE customer_id IS NOT NULL);", "reference_q_from_sql": "Which customers in the USA have no associated orders?"},
    {"id": 14, "question": "Show products with less than 10 items in stock in the 'Clothing' category.", "reference_sql": "SELECT T1.name, T1.stock_quantity FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T1.stock_quantity < 10 AND T2.name  =  'Clothing';", "reference_q_from_sql": "List clothing items with stock levels below 10."},
    {"id": 15, "question": "What is the average number of items per completed order?", "reference_sql": "SELECT AVG(item_count) FROM (SELECT order_id, SUM(quantity) as item_count FROM order_items GROUP BY order_id) AS order_counts JOIN orders o ON order_counts.order_id = o.order_id WHERE o.order_status = 'Completed';", "reference_q_from_sql": "Calculate the average quantity of items in completed orders."},
    {"id": 16, "question": "Find customers who bought product 'SuperCharger' and also reviewed it.", "reference_sql": "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id JOIN reviews r ON c.customer_id = r.customer_id AND p.product_id = r.product_id WHERE p.name = 'SuperCharger';", "reference_q_from_sql": "Identify customers who both purchased and reviewed the 'SuperCharger' product."},
    {"id": 17, "question": "List product categories that have an average product price above $200.", "reference_sql": "SELECT c.name, AVG(p.price) as avg_price FROM categories c JOIN products p ON c.category_id = p.category_id GROUP BY c.category_id, c.name HAVING AVG(p.price) > 200;", "reference_q_from_sql": "Which product categories have an average item price exceeding $200?"},
    {"id": 18, "question": "Show all details for pending orders from the last 3 days.", "reference_sql": "SELECT * FROM orders WHERE order_status = 'Pending' AND order_date >= date('now', '-3 days');", "reference_q_from_sql": "Retrieve all information for orders placed in the last 3 days that are still pending."}, # Example where SELECT * might be requested
    {"id": 19, "question": "Which customers joined in the first quarter of 2024?", "reference_sql": "SELECT name, join_date FROM customers WHERE join_date BETWEEN '2024-01-01' AND '2024-03-31';", "reference_q_from_sql": "List customers whose join date falls between January 1st and March 31st, 2024."},
    {"id": 20, "question": "Find products that have never been reviewed.", "reference_sql": "SELECT name FROM products WHERE product_id NOT IN (SELECT DISTINCT product_id FROM reviews WHERE product_id IS NOT NULL);", "reference_q_from_sql": "List products for which no reviews exist."},

    # ... Add more diverse e-commerce scenarios up to 100 ...
]
test_cases_df = pd.DataFrame(test_cases)


# --- Run the Evaluation Loop ---
results = []

if cross_encoder_qqp is None or cross_encoder_nli is None:
    print("Cannot proceed without both Cross-Encoder models. Exiting.")
else:
    print(f"\nStarting evaluation for {len(test_cases_df)} e-commerce test cases...")
    start_time_total = time.time()

    for index, row in test_cases_df.iterrows():
        original_question = row['question']
        # print(f"\n--- Processing Test Case ID: {row['id']} ---") # Reduce verbosity

        # --- Step 1: Generate SQL ---
        text_to_sql_prompt = TEXT_TO_SQL_PROMPT_TEMPLATE.format(
            schema=FULL_SCHEMA_STRING,
            user_question=original_question
        )
        try:
            predicted_sql, tts_time, tts_in_tokens, tts_out_tokens = abc_response(
                TEXT_TO_SQL_MODEL, text_to_sql_prompt
            )
        except Exception as e:
            print(f"Error in abc_response (Text-to-SQL) for ID {row['id']}: {e}")
            predicted_sql = f"ERROR_TTS: {e}"
            tts_time, tts_in_tokens, tts_out_tokens = 0, 0, 0

        # --- Step 2: Generate Question from SQL ---
        regenerated_question = "ERROR_SQL_INVALID_OR_FAILED"
        stt_time, stt_in_tokens, stt_out_tokens = 0, 0, 0
        if predicted_sql and "ERROR" not in predicted_sql:
             # Basic check for valid SQL start (can be improved)
            sql_upper = predicted_sql.strip().upper()
            if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
                 regenerated_question = "ERROR_SQL_SYNTAX_INVALID_START"
            else:
                sql_to_text_prompt = SQL_TO_TEXT_PROMPT_TEMPLATE.format(
                    schema=FULL_SCHEMA_STRING,
                    sql_query=predicted_sql
                )
                try:
                    regenerated_question, stt_time, stt_in_tokens, stt_out_tokens = abc_response(
                        SQL_TO_TEXT_MODEL, sql_to_text_prompt
                    )
                except Exception as e:
                    print(f"Error in abc_response (SQL-to-Text) for ID {row['id']}: {e}")
                    regenerated_question = f"ERROR_STT: {e}"
        else:
            regenerated_question = predicted_sql if predicted_sql else "ERROR_SQL_EMPTY"


        # --- Step 3: Calculate Similarity Scores ---
        similarity_score_qqp = -1.0 # Default/error score
        score_nli_entailment = -1.0 # Default/error score

        if original_question and regenerated_question and "ERROR" not in regenerated_question:
            sentence_pairs = [[original_question, regenerated_question]]
            try:
                # Score 1: Quora Roberta (Paraphrase/Duplicate)
                qqp_scores = cross_encoder_qqp.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
                similarity_score_qqp = qqp_scores[0]
            except Exception as e:
                print(f"Error calculating QQP similarity for ID {row['id']}: {e}")
                similarity_score_qqp = -2.0 # Indicate QQP calc error

            try:
                # Score 2: NLI DeBERTa (Entailment)
                nli_logits = cross_encoder_nli.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
                # Assuming standard MNLI label order [contradiction, neutral, entailment] -> index 2 for entailment
                # You might want to check the specific model card or experiment.
                # Using the raw logit score for entailment here. Higher means more entailment-like.
                # You could apply softmax if you prefer probabilities: probs = F.softmax(torch.tensor(nli_logits), dim=1).numpy()
                if nli_logits.shape[1] == 3:
                     score_nli_entailment = nli_logits[0][2] # Index 2 for entailment
                else:
                     print(f"Warning: NLI model output shape unexpected for ID {row['id']}. Shape: {nli_logits.shape}")
                     score_nli_entailment = -3.0 # Indicate unexpected NLI output shape
            except Exception as e:
                print(f"Error calculating NLI score for ID {row['id']}: {e}")
                score_nli_entailment = -4.0 # Indicate NLI calc error
        # else: # Keep default/error scores if regeneration failed
            # print(f"Skipping similarity calculation for ID {row['id']} due to prior errors.")


        # --- Store Results ---
        results.append({
            "test_case_id": row['id'],
            "original_question": original_question,
            # "reference_sql": row.get('reference_sql', 'N/A'),
            # "reference_q_from_sql": row.get('reference_q_from_sql', 'N/A'),
            "predicted_sql": predicted_sql,
            "tts_time_sec": round(tts_time, 3),
            "tts_input_tokens": tts_in_tokens,
            "tts_output_tokens": tts_out_tokens,
            "regenerated_question": regenerated_question,
            "stt_time_sec": round(stt_time, 3),
            "stt_input_tokens": stt_in_tokens,
            "stt_output_tokens": stt_out_tokens,
            "score_qqp_similarity": float(similarity_score_qqp),
            "score_nli_entailment": float(score_nli_entailment)
        })

    total_time = time.time() - start_time_total
    print(f"\nEvaluation finished processing {len(test_cases_df)} cases in {total_time:.2f} seconds.")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\n\n--- Evaluation Results Summary ---")
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 60) # Adjust width as needed
pd.set_option('display.width', 1000) # Adjust overall width

print(results_df[['test_case_id', 'original_question', 'predicted_sql', 'regenerated_question', 'score_qqp_similarity', 'score_nli_entailment']])

# --- Optional: Calculate Summary Statistics ---
print("\n--- Summary Statistics ---")

valid_scores_qqp = results_df[results_df['score_qqp_similarity'] >= -1.0]['score_qqp_similarity']
if not valid_scores_qqp.empty:
    print("\nQQP Similarity Scores (Higher = More Similar/Paraphrase):")
    print(f"  Average: {valid_scores_qqp.mean():.4f}")
    print(f"  Median:  {valid_scores_qqp.median():.4f}")
    print(f"  Min:     {valid_scores_qqp.min():.4f}")
    print(f"  Max:     {valid_scores_qqp.max():.4f}")
    print(f"  Std Dev: {valid_scores_qqp.std():.4f}")
else:
    print("\nNo valid QQP similarity scores were calculated.")

valid_scores_nli = results_df[results_df['score_nli_entailment'] >= -1.0]['score_nli_entailment']
if not valid_scores_nli.empty:
    print("\nNLI Entailment Scores (Higher = More Entailment):")
    print(f"  Average: {valid_scores_nli.mean():.4f}")
    print(f"  Median:  {valid_scores_nli.median():.4f}")
    print(f"  Min:     {valid_scores_nli.min():.4f}")
    print(f"  Max:     {valid_scores_nli.max():.4f}")
    print(f"  Std Dev: {valid_scores_nli.std():.4f}")
else:
    print("\nNo valid NLI entailment scores were calculated.")

# Error counts
error_tts_count = len(results_df[results_df['predicted_sql'].str.contains("ERROR", na=False)])
# Count STT errors only if TTS was not already an error
error_stt_count = len(results_df[~results_df['predicted_sql'].str.contains("ERROR", na=False) & results_df['regenerated_question'].str.contains("ERROR", na=False)])
error_qqp_count = len(results_df[results_df['score_qqp_similarity'] < -1.0])
error_nli_count = len(results_df[results_df['score_nli_entailment'] < -1.0])
print(f"\nText-to-SQL Errors: {error_tts_count}")
print(f"SQL-to-Text Errors (after successful SQL): {error_stt_count}")
print(f"QQP Similarity Calculation Errors: {error_qqp_count}")
print(f"NLI Score Calculation Errors: {error_nli_count}")


# --- Optional: Save full results to CSV ---
try:
    results_df.to_csv("ecommerce_text_sql_evaluation_results.csv", index=False)
    print("\nFull results saved to ecommerce_text_sql_evaluation_results.csv")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")
