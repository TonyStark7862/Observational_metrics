# Consolidated Text-to-SQL Evaluation Script with Few-Shot Prompts

import pandas as pd
from sentence_transformers import CrossEncoder
import torch
import time # To simulate execution time if abc_response doesn't return it
import json # For potentially parsing LLM response if it's JSON formatted

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
    # Basic simulation logic based on prompt content (can be improved)
    if "Generate SQL" in prompt:
        if "engineering department" in prompt.lower():
             simulated_response = "SELECT T1.name FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id WHERE T2.name = 'Engineering'"
        elif "average salary" in prompt.lower() and "department" in prompt.lower():
             simulated_response = "SELECT T2.name, AVG(T1.salary) FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id GROUP BY T2.name"
        elif "Project Phoenix" in prompt.lower():
             simulated_response = "SELECT T1.name FROM employees AS T1 JOIN employee_projects AS T2 ON T1.id = T2.employee_id JOIN projects AS T3 ON T2.project_id = T3.id WHERE T3.name = 'Project Phoenix'"
        elif "hierarchy" in prompt.lower():
            simulated_response = """WITH RECURSIVE EmployeeHierarchy AS (
    SELECT id, name, manager_id, 0 AS level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.id
)
SELECT name, level FROM EmployeeHierarchy ORDER BY level, name;"""
        elif "friends of friends" in prompt.lower():
             simulated_response = """WITH FriendOfFriend AS (
    SELECT f2.user2_id as fof_id
    FROM friendships f1 JOIN friendships f2 ON f1.user2_id = f2.user1_id
    WHERE f1.user1_id = 1 AND f2.user2_id != 1 AND f2.user2_id NOT IN (SELECT user2_id FROM friendships WHERE user1_id = 1)
    UNION
    SELECT f2.user1_id as fof_id
    FROM friendships f1 JOIN friendships f2 ON f1.user2_id = f2.user2_id
    WHERE f1.user1_id = 1 AND f2.user1_id != 1 AND f2.user1_id NOT IN (SELECT user2_id FROM friendships WHERE user1_id = 1)
)
SELECT DISTINCT u.name FROM users u JOIN FriendOfFriend fof ON u.id = fof.fof_id;"""
        elif "highest budget" in prompt.lower():
             simulated_response = "SELECT name FROM projects ORDER BY budget DESC LIMIT 1"
        elif "employee ID 5" in prompt.lower() and "all information" in prompt.lower():
             simulated_response = "SELECT * FROM employees WHERE id = 5"
        elif "employee ID 5" in prompt.lower() and ("name" in prompt.lower() or "names" in prompt.lower()):
              simulated_response = "SELECT name FROM employees WHERE id = 5"
        else:
            simulated_response = "SELECT column FROM table WHERE condition;" # Fallback

    elif "Generate Question" in prompt:
        if "WHERE T2.name = 'Engineering'" in prompt:
            simulated_response = "What are the names of employees belonging to the Engineering department?"
        elif "AVG(T1.salary)" in prompt and "GROUP BY T2.name" in prompt:
            simulated_response = "What is the average salary per department?"
        elif "Project Phoenix" in prompt:
            simulated_response = "List employees working on 'Project Phoenix'."
        elif "RECURSIVE EmployeeHierarchy" in prompt:
            simulated_response = "Show the employee management hierarchy with levels."
        elif "FriendOfFriend" in prompt:
             simulated_response = "Who are the friends of friends for user 1 (excluding direct friends)?"
        elif "ORDER BY budget DESC LIMIT 1" in prompt:
             simulated_response = "Which project has the highest budget?"
        elif "WHERE id = 5" in prompt and "SELECT *" in prompt:
             simulated_response = "Get all details for employee 5."
        elif "WHERE id = 5" in prompt and "SELECT name" in prompt:
              simulated_response = "What is the name of employee 5?"
        else:
             simulated_response = "What information does this SQL query provide?" # Fallback
    else:
        simulated_response = "Unknown simulation case."

    # Cleanup potential markdown/fencing if the model adds it
    if simulated_response.startswith("```sql"):
        simulated_response = simulated_response.splitlines()[1:-1] # Remove fences
        simulated_response = "\n".join(simulated_response)
    elif simulated_response.startswith("```"):
         simulated_response = simulated_response[3:-3].strip()

    simulated_time = round(0.5 + len(prompt.split()) / 500 + len(simulated_response.split()) / 100, 2)
    simulated_input_tokens = len(prompt.split())
    simulated_output_tokens = len(simulated_response.split())
    return simulated_response.strip(), simulated_time, simulated_input_tokens, simulated_output_tokens
# --- End Dummy Function ---


# --- Load Cross-Encoder Model ---
try:
    # Using quora-roberta-base, good for paraphrase/duplicate question detection
    cross_encoder_model = CrossEncoder('cross-encoder/quora-roberta-base', max_length=512, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cross-Encoder model loaded successfully onto device: {cross_encoder_model.device}")
except Exception as e:
    print(f"Error loading Cross-Encoder model: {e}")
    print("Please ensure sentence_transformers is installed ('pip install sentence_transformers') and you have an internet connection.")
    cross_encoder_model = None

# --- Define Model Identifiers ---
TEXT_TO_SQL_MODEL = 'your-text-to-sql-model-name-v1.0' # Replace
SQL_TO_TEXT_MODEL = 'your-sql-to-text-model-name-v1.0' # Replace


# --- Define Database Schema ---
SCHEMA_DICT = {
    "employees": """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY, -- Unique identifier for the employee
    name VARCHAR(100) NOT NULL, -- Full name of the employee
    department_id INTEGER, -- Foreign key referencing departments table
    manager_id INTEGER, -- Foreign key referencing employees table (self-reference for hierarchy)
    salary DECIMAL(10, 2), -- Employee's annual salary
    hire_date DATE, -- Date when the employee was hired
    is_active BOOLEAN DEFAULT TRUE -- Whether the employee is currently active
);
""",
    "departments": """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY, -- Unique identifier for the department
    name VARCHAR(100) UNIQUE NOT NULL, -- Name of the department (e.g., 'Sales', 'Engineering')
    location VARCHAR(50) -- Physical location of the department (e.g., 'New York', 'Remote')
);
""",
    "projects": """
CREATE TABLE projects (
    id INTEGER PRIMARY KEY, -- Unique identifier for the project
    name VARCHAR(150) NOT NULL, -- Name of the project
    start_date DATE, -- Project start date
    end_date DATE, -- Project end date (NULL if ongoing)
    budget DECIMAL(12, 2) -- Total budget allocated for the project
);
""",
    "employee_projects": """
CREATE TABLE employee_projects (
    employee_id INTEGER, -- Foreign key referencing employees table
    project_id INTEGER, -- Foreign key referencing projects table
    role VARCHAR(50), -- Role of the employee in the project (e.g., 'Developer', 'Manager')
    PRIMARY KEY (employee_id, project_id) -- Composite primary key
);
""",
     "users": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY, -- Unique identifier for the user
    name VARCHAR(100) NOT NULL, -- User's name
    join_date DATE -- Date user joined
);
""",
    "friendships": """
CREATE TABLE friendships (
    user1_id INTEGER, -- Foreign key referencing users table
    user2_id INTEGER, -- Foreign key referencing users table
    established_date DATE, -- Date the friendship was established
    PRIMARY KEY (user1_id, user2_id), -- Ensure unique pairs
    FOREIGN KEY (user1_id) REFERENCES users(id),
    FOREIGN KEY (user2_id) REFERENCES users(id)
);
"""
}
FULL_SCHEMA_STRING = "\n\n".join(SCHEMA_DICT.values())


# --- Define Prompt Templates with Few-Shot Examples ---

TEXT_TO_SQL_PROMPT_TEMPLATE = """Given the following database schema:
{schema}

Generate a syntactically correct SQL query for the given user question. Follow the examples provided.

--- Examples ---

Example 1:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, salary DECIMAL(10, 2));
User Question: "Which employees have a salary greater than 90000?"
SQL Query: SELECT name FROM employees WHERE salary > 90000;

Example 2:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, department_id INTEGER, salary DECIMAL(10, 2));
CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(100) UNIQUE NOT NULL);
User Question: "What is the average salary for each department?"
SQL Query: SELECT T2.name ,  AVG(T1.salary) FROM employees AS T1 JOIN departments AS T2 ON T1.department_id  =  T2.id GROUP BY T2.name;

Example 3:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL);
CREATE TABLE projects (id INTEGER PRIMARY KEY, name VARCHAR(150) NOT NULL);
CREATE TABLE employee_projects (employee_id INTEGER, project_id INTEGER, role VARCHAR(50));
User Question: "Show names of employees working on the 'Global Launch' project."
SQL Query: SELECT T1.name FROM employees AS T1 JOIN employee_projects AS T2 ON T1.id  =  T2.employee_id JOIN projects AS T3 ON T2.project_id  =  T3.id WHERE T3.name  =  'Global Launch';

Example 4:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, manager_id INTEGER);
User Question: "List the direct reports for manager ID 5."
SQL Query: SELECT name FROM employees WHERE manager_id  =  5;

--- End Examples ---

Now, generate the SQL query for this question:
User Question: "{user_question}"
SQL Query:"""


SQL_TO_TEXT_PROMPT_TEMPLATE = """Given the following database schema:
{schema}

And the following SQL query:
SQL Query: "{sql_query}"

Generate the natural language question that this SQL query most likely answers. Be concise and clear, mirroring how a user might ask. Follow the examples provided.

--- Examples ---

Example 1:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL);
SQL Query: "SELECT name FROM employees WHERE id = 101;"
Generated Question: What is the name of the employee with ID 101?

Example 2:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, department_id INTEGER);
CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(100) UNIQUE NOT NULL);
SQL Query: "SELECT T2.name ,  count(*) FROM employees AS T1 JOIN departments AS T2 ON T1.department_id  =  T2.id GROUP BY T2.name;"
Generated Question: How many employees are in each department?

Example 3:
Relevant Schema:
CREATE TABLE projects (id INTEGER PRIMARY KEY, name VARCHAR(150) NOT NULL, budget DECIMAL(12, 2));
SQL Query: "SELECT name FROM projects ORDER BY budget ASC LIMIT 1;"
Generated Question: Which project has the smallest budget?

Example 4:
Relevant Schema:
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, manager_id INTEGER);
SQL Query: "WITH RECURSIVE Subordinates AS (SELECT id FROM employees WHERE id = 5 UNION ALL SELECT e.id FROM employees e INNER JOIN Subordinates s ON e.manager_id = s.id) SELECT count(*) FROM Subordinates WHERE id != 5;"
Generated Question: How many employees report (directly or indirectly) to employee ID 5?

--- End Examples ---

Now, generate the question for this query:
SQL Query: "{sql_query}"
Generated Question:"""


# --- Define Test Cases (Sample - Expand to 100+) ---
test_cases = [
    # Using IDs from previous examples for consistency where applicable
    {"id": 1, "question": "Show the names of all employees in the Engineering department.", "reference_sql": "SELECT T1.name FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id WHERE T2.name = 'Engineering'", "reference_q_from_sql": "What are the names of employees belonging to the Engineering department?"},
    {"id": 2, "question": "Which employees have a salary of 85000 or more?", "reference_sql": "SELECT name FROM employees WHERE salary >= 85000", "reference_q_from_sql": "List the names of employees whose salary is greater than or equal to 85000."},
    {"id": 3, "question": "List active employees hired after January 1st, 2023.", "reference_sql": "SELECT name FROM employees WHERE hire_date > '2023-01-01' AND is_active = TRUE", "reference_q_from_sql": "What are the names of active employees hired subsequent to 2023-01-01?"},
    {"id": 4, "question": "What are the names of employees working on the 'Project Phoenix' project?", "reference_sql": "SELECT T1.name FROM employees AS T1 JOIN employee_projects AS T2 ON T1.id = T2.employee_id JOIN projects AS T3 ON T2.project_id = T3.id WHERE T3.name = 'Project Phoenix'", "reference_q_from_sql": "List employee names associated with the project named 'Project Phoenix'."},
    {"id": 5, "question": "Show the department name for the employee named 'Alice Wonderland'.", "reference_sql": "SELECT T2.name FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id WHERE T1.name = 'Alice Wonderland'", "reference_q_from_sql": "What department does 'Alice Wonderland' work in?"},
    {"id": 6, "question": "How many employees are there in total?", "reference_sql": "SELECT count(*) FROM employees", "reference_q_from_sql": "What is the total count of employees?"},
    {"id": 7, "question": "What is the average salary in the Sales department?", "reference_sql": "SELECT avg(T1.salary) FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id WHERE T2.name = 'Sales'", "reference_q_from_sql": "Calculate the average salary for employees in the Sales department."},
    {"id": 8, "question": "Find the maximum budget among all projects.", "reference_sql": "SELECT max(budget) FROM projects", "reference_q_from_sql": "What is the highest project budget?"},
    {"id": 9, "question": "Count the number of employees in each department.", "reference_sql": "SELECT T2.name ,  count(*) FROM employees AS T1 JOIN departments AS T2 ON T1.department_id  =  T2.id GROUP BY T2.name", "reference_q_from_sql": "Provide a count of employees grouped by department name."},
    {"id": 10, "question": "List the 5 most recently hired employees.", "reference_sql": "SELECT name FROM employees ORDER BY hire_date DESC LIMIT 5", "reference_q_from_sql": "Who are the 5 employees with the latest hire dates?"},
    {"id": 11, "question": "Which project has the highest budget? Show its name.", "reference_sql": "SELECT name FROM projects ORDER BY budget DESC LIMIT 1", "reference_q_from_sql": "What is the name of the project with the largest budget?"},
    {"id": 12, "question": "Show all information about the employee with ID 5.", "reference_sql": "SELECT * FROM employees WHERE id = 5", "reference_q_from_sql": "Retrieve all details for the employee whose ID is 5."},
    {"id": 13, "question": "Just give me the names for employee ID 5.", "reference_sql": "SELECT name FROM employees WHERE id = 5", "reference_q_from_sql": "What is the name of the employee with ID 5?"},
    {"id": 14, "question": "Are there any ongoing projects?", "reference_sql": "SELECT count(*) FROM projects WHERE end_date IS NULL", "reference_q_from_sql": "How many projects currently have no end date?"},
    {"id": 15, "question": "List employees who are not assigned to any project.", "reference_sql": "SELECT name FROM employees WHERE id NOT IN (SELECT DISTINCT employee_id FROM employee_projects)", "reference_q_from_sql": "Which employees are not listed in the employee_projects table?"},
    {"id": 16, "question": "What are the different locations where departments are based?", "reference_sql": "SELECT DISTINCT location FROM departments WHERE location IS NOT NULL", "reference_q_from_sql": "List the unique non-null department locations."},
    {"id": 17, "question": "List all employees managed by the employee with ID 2.", "reference_sql": "SELECT name FROM employees WHERE manager_id = 2", "reference_q_from_sql": "Who reports directly to the manager with ID 2?"},
    {"id": 18, "question": "Show the management hierarchy starting from the top (employees with no manager). List name and level.", "reference_sql": """WITH RECURSIVE EmployeeHierarchy AS (SELECT id, name, manager_id, 0 AS level FROM employees WHERE manager_id IS NULL UNION ALL SELECT e.id, e.name, e.manager_id, eh.level + 1 FROM employees e INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.id) SELECT name, level FROM EmployeeHierarchy ORDER BY level, name;""", "reference_q_from_sql": "Display the employee hierarchy with names and levels, starting from top-level managers."},
    {"id": 19, "question": "Find the direct friends of user ID 1.", "reference_sql": "SELECT T2.name FROM friendships AS T1 JOIN users AS T2 ON T1.user2_id  =  T2.id WHERE T1.user1_id = 1", "reference_q_from_sql": "List the names of users who are friends with user ID 1."},
    {"id": 20, "question": "Find the friends of friends for user ID 1 (excluding direct friends).", "reference_sql": """WITH FriendOfFriend AS (SELECT f2.user2_id as fof_id FROM friendships f1 JOIN friendships f2 ON f1.user2_id = f2.user1_id WHERE f1.user1_id = 1 AND f2.user2_id != 1 AND f2.user2_id NOT IN (SELECT user2_id FROM friendships WHERE user1_id = 1) UNION SELECT f2.user1_id as fof_id FROM friendships f1 JOIN friendships f2 ON f1.user2_id = f2.user2_id WHERE f1.user1_id = 1 AND f2.user1_id != 1 AND f2.user1_id NOT IN (SELECT user2_id FROM friendships WHERE user1_id = 1)) SELECT DISTINCT u.name FROM users u JOIN FriendOfFriend fof ON u.id = fof.fof_id;""", "reference_q_from_sql": "Who are the friends of user 1's friends, but not direct friends of user 1?"},
    # Add more cases up to 100...
]

test_cases_df = pd.DataFrame(test_cases)


# --- Run the Evaluation Loop ---
results = []

if cross_encoder_model is None:
    print("Cannot proceed without Cross-Encoder model. Exiting.")
else:
    print(f"\nStarting evaluation for {len(test_cases_df)} test cases...")
    start_time_total = time.time()

    for index, row in test_cases_df.iterrows():
        # print(f"\n--- Processing Test Case ID: {row['id']} ---") # Reduce verbosity
        original_question = row['question']

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
            if not predicted_sql.strip().upper().startswith(('SELECT', 'WITH')):
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
            # Propagate error or handle empty SQL
             regenerated_question = predicted_sql if predicted_sql else "ERROR_SQL_EMPTY"


        # --- Step 3: Calculate Similarity Score ---
        similarity_score = -1.0 # Default/error score
        if original_question and regenerated_question and "ERROR" not in regenerated_question:
             try:
                 sentence_pairs = [[original_question, regenerated_question]]
                 # The predict method might take some time, especially for the first few calls
                 score_calculation_start = time.time()
                 scores = cross_encoder_model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
                 score_calculation_time = time.time() - score_calculation_start
                 similarity_score = scores[0]
                 # print(f"Similarity Score (ID: {row['id']}): {similarity_score:.4f} (calc time: {score_calculation_time:.2f}s)") # Verbose
             except Exception as e:
                print(f"Error calculating similarity for ID {row['id']}: {e}")
                similarity_score = -2.0 # Indicate similarity calc error specifically
        # else: # Keep default/error score if regeneration failed
            # print(f"Skipping similarity calculation for ID {row['id']} due to prior errors.")


        # --- Store Results ---
        results.append({
            "test_case_id": row['id'],
            "original_question": original_question,
            # "reference_sql": row.get('reference_sql', 'N/A'), # Optional for final report
            # "reference_q_from_sql": row.get('reference_q_from_sql', 'N/A'), # Optional
            "predicted_sql": predicted_sql,
            "tts_time_sec": round(tts_time, 3),
            "tts_input_tokens": tts_in_tokens,
            "tts_output_tokens": tts_out_tokens,
            "regenerated_question": regenerated_question,
            "stt_time_sec": round(stt_time, 3),
            "stt_input_tokens": stt_in_tokens,
            "stt_output_tokens": stt_out_tokens,
            "similarity_score": float(similarity_score)
        })

    total_time = time.time() - start_time_total
    print(f"\nEvaluation finished processing {len(test_cases_df)} cases in {total_time:.2f} seconds.")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\n\n--- Evaluation Results Summary ---")
# Display options for better readability
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100) # Show more text
print(results_df[['test_case_id', 'original_question', 'predicted_sql', 'regenerated_question', 'similarity_score', 'tts_time_sec', 'stt_time_sec']])

# --- Optional: Calculate Summary Statistics ---
print("\n--- Summary Statistics ---")
valid_scores = results_df[results_df['similarity_score'] >= -1.0]['similarity_score'] # Exclude calculation errors
if not valid_scores.empty:
    print(f"Average Similarity Score (valid runs): {valid_scores.mean():.4f}")
    print(f"Median Similarity Score (valid runs): {valid_scores.median():.4f}")
    print(f"Min Similarity Score (valid runs): {valid_scores.min():.4f}")
    print(f"Max Similarity Score (valid runs): {valid_scores.max():.4f}")
    print(f"Standard Deviation (valid runs): {valid_scores.std():.4f}")
else:
    print("No valid similarity scores were calculated.")

error_tts_count = len(results_df[results_df['predicted_sql'].str.contains("ERROR", na=False)])
error_stt_count = len(results_df[results_df['regenerated_question'].str.contains("ERROR", na=False)])
error_sim_count = len(results_df[results_df['similarity_score'] < -1.0])
print(f"\nText-to-SQL Errors: {error_tts_count}")
print(f"SQL-to-Text Errors: {error_stt_count - error_tts_count}") # Avoid double counting
print(f"Similarity Calculation Errors: {error_sim_count}")


# --- Optional: Save full results to CSV ---
try:
    results_df.to_csv("text_sql_evaluation_full_results.csv", index=False)
    print("\nFull results saved to text_sql_evaluation_full_results.csv")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")
