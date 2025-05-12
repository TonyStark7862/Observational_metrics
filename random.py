# Consolidated Text-to-SQL Evaluation Script (E-commerce)
# V3: Generates multiple SQL-to-Text questions, uses max QQP similarity,
#     and converts NLI logits to labels.

import pandas as pd
from sentence_transformers import CrossEncoder
import torch
import torch.nn.functional as F # For softmax
import time
import json # For parsing potential JSON list response
import numpy as np # For argmax

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
        # (Same SQL generation simulation logic as before)
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
            simulated_response = "SELECT order_id, order_date, total_amount FROM orders WHERE order_status = 'Pending' AND order_date >= date('now', '-7 days');"
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

    elif "Generate 3 distinct" in prompt: # Updated prompt check for SQL-to-Text
        # Simulate returning a JSON list of questions
        q_list = []
        base_q = "Placeholder question"
        if "city = 'London'" in prompt:
            base_q = "List customers residing in London."
            q_list = [base_q, "Show me customers whose city is London.", "Which customers live in London?"]
        elif "AVG(total_amount)" in prompt:
             base_q = "What is the average value of completed orders?"
             q_list = [base_q, "Calculate the mean total amount for orders marked completed.", "Find the average completed order total."]
        elif "ORDER BY total_sales DESC" in prompt:
             base_q = "Which 5 products generated the most sales revenue from completed orders?"
             q_list = ["List the top 5 products by total sales.", base_q, "Show the 5 products with highest sales value."]
        elif "order_status = 'Pending'" in prompt and "date('now', '-7 days')" in prompt:
             base_q = "Show pending orders from the last week."
             q_list = [base_q, "List orders with status 'Pending' placed in the last 7 days.", "What are the recent pending orders (last 7 days)?"]
        elif "'Widget X'" in prompt and "'Widget Y'" in prompt:
            base_q = "Which customers purchased both 'Widget X' and 'Widget Y'?"
            q_list = [base_q, "Find customers who bought 'Widget X' and also 'Widget Y'.", "List customers associated with orders containing both 'Widget X' and 'Widget Y'."]
        elif "rating < 3" in prompt:
            base_q = "Show review text for products with ratings below 3 stars."
            q_list = [base_q, "Display reviews and product names where the rating is less than 3.", "Which products received poor reviews (under 3 stars)?"]
        elif "c.name = 'Electronics'" in prompt and "BETWEEN 500 AND 1000" in prompt:
             base_q = "List electronic products priced between 500 and 1000."
             q_list = [base_q, "Show electronics category items costing from 500 to 1000.", "Find electronic products in the 500-1000 price range."]
        else:
             base_q = "What are the first 10 products?" # Fallback
             q_list = [base_q, "Show the names of the initial 10 products.", "List top 10 products."]

        simulated_response = json.dumps(q_list) # Return as JSON string

    else:
        simulated_response = "Unknown simulation case."

    # Cleanup potential markdown/fencing
    if '```sql' in simulated_response:
        simulated_response = simulated_response.split('```sql', 1)[1].split('```', 1)[0]
    elif '```json' in simulated_response:
         simulated_response = simulated_response.split('```json', 1)[1].split('```', 1)[0]
    elif simulated_response.startswith("```"):
         simulated_response = simulated_response[3:-3].strip()

    simulated_time = round(0.6 + len(prompt.split()) / 400 + len(simulated_response.split()) / 80, 2) # Slightly longer for multi-question
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
    # Define the expected label order for the NLI model
    # Common order for MNLI fine-tuned models: Contradiction, Neutral, Entailment
    nli_label_mapping = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    print("Cross-Encoder NLI model loaded successfully.")
except Exception as e:
    print(f"Error loading Cross-Encoder NLI model: {e}")

# --- Define Model Identifiers ---
TEXT_TO_SQL_MODEL = 'your-text-to-sql-model-ecommerce-v1.0' # Replace
SQL_TO_TEXT_MODEL = 'your-sql-to-text-model-ecommerce-v1.0' # Replace


# --- Define E-commerce Database Schema ---
SCHEMA_DICT = {
    "customers": """
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(100) UNIQUE, phone VARCHAR(20), address VARCHAR(255), city VARCHAR(50), country VARCHAR(50), join_date DATE); -- Info about customers
""",
    "categories": """
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE, description TEXT); -- Product categories like Electronics, Books, Clothing
""",
    "products": """
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150) NOT NULL, description TEXT, price DECIMAL(10, 2) NOT NULL, category_id INTEGER, stock_quantity INTEGER DEFAULT 0, average_rating DECIMAL(3, 2), FOREIGN KEY (category_id) REFERENCES categories(category_id)); -- Product details
""",
    "orders": """
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, order_status VARCHAR(20) NOT NULL DEFAULT 'Pending', total_amount DECIMAL(12, 2), shipping_address VARCHAR(255), FOREIGN KEY (customer_id) REFERENCES customers(customer_id)); -- Customer order header
""",
    "order_items": """
CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER NOT NULL, price_per_unit DECIMAL(10, 2) NOT NULL, FOREIGN KEY (order_id) REFERENCES orders(order_id), FOREIGN KEY (product_id) REFERENCES products(product_id)); -- Items within an order
""",
    "reviews": """
CREATE TABLE reviews (review_id INTEGER PRIMARY KEY, product_id INTEGER, customer_id INTEGER, rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5), review_text TEXT, review_date DATE DEFAULT CURRENT_DATE, FOREIGN KEY (product_id) REFERENCES products(product_id), FOREIGN KEY (customer_id) REFERENCES customers(customer_id)); -- Customer reviews for products
"""
}
FULL_SCHEMA_STRING = "-- E-commerce Database Schema --\n\n" + "\n\n".join(SCHEMA_DICT.values())


# --- Define Prompt Templates with Few-Shot Examples (Updated SQL-to-Text) ---

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

Generate 3 distinct but accurate natural language questions that this SQL query precisely answers. Output the questions as a JSON list of strings. Be concise and clear, mirroring how different users might ask. Follow the examples provided.

--- Examples ---

Example 1:
Relevant Schema:
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150), price DECIMAL(10, 2));
SQL Query: "SELECT name FROM products WHERE price < 50.00;"
Generated Questions: ["Which products cost less than $50?", "List product names with a price under 50 dollars.", "Show me products cheaper than $50."]

Example 2:
Relevant Schema:
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_status VARCHAR(20));
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100));
SQL Query: "SELECT T2.name FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id WHERE T1.order_status  =  'Pending';"
Generated Questions: ["List the names of customers who have pending orders.", "Which customers currently have orders with 'Pending' status?", "Show customer names for all pending orders."]

Example 3:
Relevant Schema:
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150), average_rating DECIMAL(3, 2));
SQL Query: "SELECT name FROM products ORDER BY average_rating DESC LIMIT 3;"
Generated Questions: ["What are the top 3 highest-rated products?", "List the names of the three products with the best average rating.", "Show the 3 products rated highest by customers."]

Example 4:
Relevant Schema:
CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(150));
SQL Query: "SELECT T2.name ,  SUM(T1.quantity) FROM order_items AS T1 JOIN products AS T2 ON T1.product_id  =  T2.product_id GROUP BY T2.name HAVING SUM(T1.quantity) > 100;"
Generated Questions: ["Which products have sold more than 100 units in total?", "List products with total sales quantity exceeding 100 units.", "Show product names where the sum of quantities ordered is over 100."]

--- End Examples ---

Now, generate the 3 questions as a JSON list for this query:
SQL Query: "{sql_query}"
Generated Questions:"""


# --- Define E-commerce Test Cases (Sample - Expand to 100+) ---
test_cases = [
    # Using IDs from previous examples for consistency where applicable
    {"id": 1, "question": "Find customer details for email 'test@example.com'.", "reference_sql": "SELECT * FROM customers WHERE email = 'test@example.com';", "reference_q_from_sql": "Show all information for the customer with email 'test@example.com'."},
    {"id": 2, "question": "List products that cost more than $1000.", "reference_sql": "SELECT name, price FROM products WHERE price > 1000.00;", "reference_q_from_sql": "What are the names and prices of products exceeding $1000?"},
    {"id": 3, "question": "Show orders placed on May 1st, 2025.", "reference_sql": "SELECT order_id, customer_id, total_amount FROM orders WHERE date(order_date) = '2025-05-01';", "reference_q_from_sql": "Which orders were created on 2025-05-01?"},
    {"id": 4, "question": "Find reviews with a 5-star rating.", "reference_sql": "SELECT review_id, product_id, customer_id, review_text FROM reviews WHERE rating = 5;", "reference_q_from_sql": "Show the reviews that gave a 5-star rating."},
    {"id": 5, "question": "What products are in the 'Electronics' category?", "reference_sql": "SELECT T1.name FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T2.name  =  'Electronics';", "reference_q_from_sql": "List product names belonging to the 'Electronics' category."},
    {"id": 6, "question": "Show the order dates for orders placed by customer ID 42.", "reference_sql": "SELECT order_date FROM orders WHERE customer_id = 42;", "reference_q_from_sql": "What are the dates of orders made by customer 42?"},
    {"id": 7, "question": "List the products included in order ID 101.", "reference_sql": "SELECT T2.name, T1.quantity, T1.price_per_unit FROM order_items AS T1 JOIN products AS T2 ON T1.product_id = T2.product_id WHERE T1.order_id = 101;", "reference_q_from_sql": "What products (name, quantity, price) were part of order 101?"},
    {"id": 8, "question": "Find the names of customers who reviewed the product with ID 50.", "reference_sql": "SELECT T2.name FROM reviews AS T1 JOIN customers AS T2 ON T1.customer_id = T2.customer_id WHERE T1.product_id = 50;", "reference_q_from_sql": "Which customers wrote a review for product 50?"},
    {"id": 9, "question": "How many products are in the 'Books' category?", "reference_sql": "SELECT count(*) FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T2.name  =  'Books';", "reference_q_from_sql": "What is the total number of products classified under 'Books'?"},
    {"id": 10, "question": "What is the total value of all 'Completed' orders?", "reference_sql": "SELECT SUM(total_amount) FROM orders WHERE order_status = 'Completed';", "reference_q_from_sql": "Calculate the sum of total amounts for all completed orders."},
    {"id": 11, "question": "Find the average rating for product ID 75.", "reference_sql": "SELECT AVG(rating) FROM reviews WHERE product_id = 75;", "reference_q_from_sql": "What is the average customer rating given to product 75?"},
    {"id": 12, "question": "How many orders has each customer placed? Show customer name and count.", "reference_sql": "SELECT T2.name ,  count(T1.order_id) FROM orders AS T1 JOIN customers AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T2.name ORDER BY count(T1.order_id) DESC;", "reference_q_from_sql": "Count the number of orders per customer, showing their name."},
    {"id": 13, "question": "List customers from 'USA' who haven't placed any orders.", "reference_sql": "SELECT name FROM customers WHERE country = 'USA' AND customer_id NOT IN (SELECT DISTINCT customer_id FROM orders WHERE customer_id IS NOT NULL);", "reference_q_from_sql": "Which customers in the USA have no associated orders?"},
    {"id": 14, "question": "Show products with less than 10 items in stock in the 'Clothing' category.", "reference_sql": "SELECT T1.name, T1.stock_quantity FROM products AS T1 JOIN categories AS T2 ON T1.category_id  =  T2.category_id WHERE T1.stock_quantity < 10 AND T2.name  =  'Clothing';", "reference_q_from_sql": "List clothing items with stock levels below 10."},
    {"id": 15, "question": "What is the average number of items per completed order?", "reference_sql": "SELECT AVG(item_count) FROM (SELECT order_id, SUM(quantity) as item_count FROM order_items GROUP BY order_id) AS order_counts JOIN orders o ON order_counts.order_id = o.order_id WHERE o.order_status = 'Completed';", "reference_q_from_sql": "Calculate the average quantity of items in completed orders."},
    {"id": 16, "question": "Find customers who bought product 'SuperCharger' and also reviewed it.", "reference_sql": "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id JOIN reviews r ON c.customer_id = r.customer_id AND p.product_id = r.product_id WHERE p.name = 'SuperCharger';", "reference_q_from_sql": "Identify customers who both purchased and reviewed the 'SuperCharger' product."},
    {"id": 17, "question": "List product categories that have an average product price above $200.", "reference_sql": "SELECT c.name, AVG(p.price) as avg_price FROM categories c JOIN products p ON c.category_id = p.category_id GROUP BY c.category_id, c.name HAVING AVG(p.price) > 200;", "reference_q_from_sql": "Which product categories have an average item price exceeding $200?"},
    {"id": 18, "question": "Show all details for pending orders from the last 3 days.", "reference_sql": "SELECT * FROM orders WHERE order_status = 'Pending' AND order_date >= date('now', '-3 days');", "reference_q_from_sql": "Retrieve all information for orders placed in the last 3 days that are still pending."}, # Example where SELECT * might be requested
    {"id": 19, "question": "Which customers joined in the first quarter of 2024?", "reference_sql": "SELECT name, join_date FROM customers WHERE join_date BETWEEN '2024-01-01' AND '2024-03-31';", "reference_q_from_sql": "List customers whose join date falls between January 1st and March 31st, 2024."},
    {"id": 20, "question": "Find products that have never been reviewed.", "reference_sql": "SELECT name FROM products WHERE product_id NOT IN (SELECT DISTINCT product_id FROM reviews WHERE product_id IS NOT NULL);", "reference_q_from_sql": "List products for which no reviews exist."},
]
test_cases_df = pd.DataFrame(test_cases)


# --- Run the Evaluation Loop ---
results = []
nli_label_mapping = {0: 'contradiction', 1: 'neutral', 2: 'entailment'} # Standard MNLI mapping

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

        # --- Step 2: Generate Multiple Questions from SQL ---
        regenerated_question_list_str = "ERROR_SQL_INVALID_OR_FAILED"
        regenerated_question_list = []
        stt_time, stt_in_tokens, stt_out_tokens = 0, 0, 0
        best_q_regen = None # Track the question with the highest QQP score

        if predicted_sql and "ERROR" not in predicted_sql:
            sql_upper = predicted_sql.strip().upper()
            if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
                 regenerated_question_list_str = "ERROR_SQL_SYNTAX_INVALID_START"
            else:
                sql_to_text_prompt = SQL_TO_TEXT_PROMPT_TEMPLATE.format(
                    schema=FULL_SCHEMA_STRING,
                    sql_query=predicted_sql
                )
                try:
                    # Expecting a JSON list string from the LLM now
                    response_text, stt_time, stt_in_tokens, stt_out_tokens = abc_response(
                        SQL_TO_TEXT_MODEL, sql_to_text_prompt
                    )
                    regenerated_question_list_str = response_text # Store the raw response
                    try:
                        # Attempt to parse the JSON list
                        regenerated_question_list = json.loads(response_text)
                        if not isinstance(regenerated_question_list, list):
                             print(f"Warning: SQL-to-Text response for ID {row['id']} not a list after JSON parse.")
                             regenerated_question_list_str = "ERROR_STT_NOT_A_LIST"
                             regenerated_question_list = []

                    except json.JSONDecodeError as json_e:
                         print(f"Warning: SQL-to-Text response for ID {row['id']} not valid JSON: {json_e}")
                         # Fallback: try to treat the raw response as a single question
                         regenerated_question_list = [response_text]
                         regenerated_question_list_str = f"ERROR_STT_JSON_PARSE_FALLBACK: {response_text}" # Mark as fallback
                    except Exception as parse_e:
                         print(f"Error processing STT response for ID {row['id']}: {parse_e}")
                         regenerated_question_list_str = f"ERROR_STT_PROCESSING: {parse_e}"
                         regenerated_question_list = []

                except Exception as e:
                    print(f"Error in abc_response (SQL-to-Text) for ID {row['id']}: {e}")
                    regenerated_question_list_str = f"ERROR_STT_API: {e}"
                    regenerated_question_list = []
        else:
            regenerated_question_list_str = predicted_sql if predicted_sql else "ERROR_SQL_EMPTY"
            regenerated_question_list = []


        # --- Step 3: Calculate Max QQP Similarity & NLI Label for Best QQP Match ---
        similarity_score_qqp_max = -1.0 # Default/error score
        nli_predicted_label = "ERROR"

        if original_question and regenerated_question_list:
            max_score_found = -float('inf')
            best_q_regen_for_nli = None

            try:
                # Calculate QQP score for each regenerated question
                sentence_pairs_qqp = [[original_question, q_regen] for q_regen in regenerated_question_list]
                qqp_scores = cross_encoder_qqp.predict(sentence_pairs_qqp, convert_to_numpy=True, show_progress_bar=False)

                if len(qqp_scores) > 0:
                    max_score_found = np.max(qqp_scores)
                    best_q_index = np.argmax(qqp_scores)
                    best_q_regen_for_nli = regenerated_question_list[best_q_index]
                    similarity_score_qqp_max = max_score_found

            except Exception as e:
                print(f"Error calculating QQP similarity for ID {row['id']}: {e}")
                similarity_score_qqp_max = -2.0 # Indicate QQP calc error

            # Calculate NLI score only for the best QQP match, if found
            if best_q_regen_for_nli is not None:
                try:
                    sentence_pair_nli = [[original_question, best_q_regen_for_nli]]
                    nli_logits = cross_encoder_nli.predict(sentence_pair_nli, convert_to_numpy=True, show_progress_bar=False)

                    if nli_logits.shape[1] == len(nli_label_mapping):
                        # Apply softmax to get probabilities (optional, but good practice)
                        # probs = F.softmax(torch.tensor(nli_logits), dim=1).numpy()[0]
                        predicted_index = np.argmax(nli_logits[0])
                        nli_predicted_label = nli_label_mapping.get(predicted_index, "Unknown Index")
                    else:
                         print(f"Warning: NLI model output shape unexpected for ID {row['id']}. Shape: {nli_logits.shape}")
                         nli_predicted_label = "ERROR_NLI_SHAPE"
                except Exception as e:
                    print(f"Error calculating NLI score for ID {row['id']}: {e}")
                    nli_predicted_label = "ERROR_NLI_CALC"
            else:
                 nli_predicted_label = "ERROR_NO_BEST_Q" # If no valid QQP score was found

        # --- Store Results ---
        results.append({
            "test_case_id": row['id'],
            "original_question": original_question,
            "predicted_sql": predicted_sql,
            "tts_time_sec": round(tts_time, 3),
            "tts_input_tokens": tts_in_tokens,
            "tts_output_tokens": tts_out_tokens,
            "regenerated_question_list": regenerated_question_list_str, # Store raw response or error
            "stt_time_sec": round(stt_time, 3),
            "stt_input_tokens": stt_in_tokens,
            "stt_output_tokens": stt_out_tokens,
            "score_qqp_similarity_max": float(similarity_score_qqp_max),
            "nli_predicted_label": nli_predicted_label
        })

    total_time = time.time() - start_time_total
    print(f"\nEvaluation finished processing {len(test_cases_df)} cases in {total_time:.2f} seconds.")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\n\n--- Evaluation Results Summary ---")
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 40) # Adjust width
pd.set_option('display.width', 1200) # Adjust overall width

print(results_df[['test_case_id', 'original_question', 'predicted_sql', 'regenerated_question_list', 'score_qqp_similarity_max', 'nli_predicted_label']])

# --- Optional: Calculate Summary Statistics ---
print("\n--- Summary Statistics ---")

valid_scores_qqp = results_df[results_df['score_qqp_similarity_max'] >= -1.0]['score_qqp_similarity_max']
if not valid_scores_qqp.empty:
    print("\nMax QQP Similarity Scores (Higher = More Similar/Paraphrase Found):")
    print(f"  Average: {valid_scores_qqp.mean():.4f}")
    print(f"  Median:  {valid_scores_qqp.median():.4f}")
    print(f"  Min:     {valid_scores_qqp.min():.4f}")
    print(f"  Max:     {valid_scores_qqp.max():.4f}")
    print(f"  Std Dev: {valid_scores_qqp.std():.4f}")
else:
    print("\nNo valid QQP similarity scores were calculated.")

print("\nNLI Predicted Label Distribution:")
if 'nli_predicted_label' in results_df.columns:
    print(results_df['nli_predicted_label'].value_counts())
else:
    print("NLI Label column not found.")

# Error counts
error_tts_count = len(results_df[results_df['predicted_sql'].str.contains("ERROR", na=False)])
error_stt_count = len(results_df[~results_df['predicted_sql'].str.contains("ERROR", na=False) & results_df['regenerated_question_list'].str.contains("ERROR", na=False)])
error_qqp_count = len(results_df[results_df['score_qqp_similarity_max'] < -1.0])
error_nli_count = len(results_df[~results_df['nli_predicted_label'].isin(nli_label_mapping.values()) & ~results_df['nli_predicted_label'].str.contains("NO_BEST_Q", na=False)]) # Count errors not including valid labels or known failure modes

print(f"\nText-to-SQL Errors: {error_tts_count}")
print(f"SQL-to-Text Errors (after successful SQL): {error_stt_count}")
print(f"QQP Similarity Calculation Errors: {error_qqp_count}")
print(f"NLI Label Calculation Errors: {error_nli_count}")


# --- Optional: Save full results to CSV ---
try:
    results_df.to_csv("ecommerce_text_sql_eval_v3_results.csv", index=False)
    print("\nFull results saved to ecommerce_text_sql_eval_v3_results.csv")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")
