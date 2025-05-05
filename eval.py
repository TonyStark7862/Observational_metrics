import pandas as pd
import sqlite3

actual_data_1 = {
    'id_1': [1, 2, 3, 4, 5, 6],
    'name_1': ['John', 'Alice', 'Bob', 'Emily', 'Charlie', 'AE3'],
    'age_1': [28, 32, 25, 35, 29, 30],
    'city_1': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Boston', 'Bay'],
    'salary_1': [75000, 80000, 70000, 90000, 65000, 95000],
    'department_1': ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Payroll']
}

df_1 = pd.DataFrame(actual_data_1)

actual_data_2 = {
    'id_2': [1, 2, 3, 4, 5],
    'name_2': ['Mike', 'Eva', 'David', 'Sophia', 'Frank'],
    'age_2': [32, 28, 30, 27, 35],
    'city_2': ['Chicago', 'New York', 'Los Angeles', 'San Francisco', 'Boston'],
    'salary_2': [85000, 90000, 80000, 95000, 88000],
    'department_2': ['IT', 'Sales', 'Marketing', 'HR', 'Finance']
}

df_2 = pd.DataFrame(actual_data_2)

predicted_queries = [
    'SELECT * FROM df_1 WHERE age_1 > 25',
    'SELECT name_2 FROM df_2 WHERE age_2 > 25',
    'SELECT name_2 FROM df_2 WHERE age_2>25',
    'SELECT name_1 FROM df_1 WHERE city_1 = "Chicago"',
    'SELECT name_1, age_1 FROM df_1 WHERE city_1 = "Chicago" and salary_1>70000',
    'SELECT * FROM df_1 WHERE name_1 like "A%"',
    'SELECT * FROM df_1 WHERE name_1 like "%A%"'
]

true_queries = [
    'SELECT * FROM df_1 WHERE age_1 > 25',
    'SELECT * FROM df_2 WHERE age_2 > 25',
    'SELECT name_2 FROM df_2 WHERE age_2 > 25',
    'SELECT name_1, age_1 FROM df_1 WHERE city_1 = "New York"',
    'SELECT name_1, age_1 FROM df_1 WHERE salary_1>70000 and city_1 = "Chicago"',
    'SELECT * FROM df_1 WHERE name_1 like "A%"',
    'SELECT * FROM df_1 WHERE name_1 like "%AE%"'
]

conn = sqlite3.connect(':memory:')

df_1.to_sql('df_1', conn, index=False, if_exists='replace')
df_2.to_sql('df_2', conn, index=False, if_exists='replace')

def check_results_similarity(predicted_queries, true_queries):
    similarity_list = []
    
    for predicted_query, true_query in zip(predicted_queries, true_queries):
        original_predicted_query = predicted_query
        original_true_query = true_query
        
        # If one query uses '*' and the other uses specific column names
        if '*' in predicted_query and '*' not in true_query:
            # Use column names from the true query in place of '*'
            true_columns = [col.strip() for col in true_query.split('SELECT')[1].split('FROM')[0].split(',')]
            predicted_query = predicted_query.replace('*', ', '.join(true_columns))
        elif '*' in true_query and '*' not in predicted_query:
            # Use column names from the predicted query in place of '*'
            predicted_columns = [col.strip() for col in predicted_query.split('SELECT')[1].split('FROM')[0].split(',')]
            true_query = true_query.replace('*', ', '.join(predicted_columns))
            
        # Execute the SQL queries on the respective DataFrames
        predicted_result_df = pd.read_sql_query(predicted_query, conn)
        true_result_df = pd.read_sql_query(true_query, conn)
        
        # Check if results are similar for each pair of queries
        similarity = 'Yes' if predicted_result_df.equals(true_result_df) else 'No'
        similarity_list.append(similarity)
        
        print(f"Results for Predicted Query:\n{original_predicted_query}\n\nand\nTrue Query:\n{original_true_query}\nare {similarity}\n\n")
    
    return similarity_list

similarity_result = check_results_similarity(predicted_queries, true_queries)

conn.close()

result_df = pd.DataFrame({'predicted_queries': predicted_queries, 'true_queries': true_queries, 'similarity': similarity_result})

result_df.to_csv('result.csv', index=False)
