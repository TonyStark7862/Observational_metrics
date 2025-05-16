import re
import logging # Logging setup remains commented as in original code
from sql_metadata import Parser


# --- Centralized Configuration Section ---
# This default mapping will be used if no CREATE statements are provided
# Keys = Table names (as expected from sql-metadata parser, often lowercase)
# Values = List of allowed column names (use lowercase for consistency)
TABLE_COLUMN_MAPPING = {
    "ecommerce_product_detail": ['*', 'product_id', 'count', 'user_id', 'category', 'department', 'item_id', 'assigned_to', 'tags', 'priority', 'comments', 'remediation_owner'],
    "ecommerce_product_metadata": ['*', 'product_id', 'dataset_definition_id', 'query_details', 'product_name', 'sql_filter', 'product_metadata', 'category', 'comments', 'priority', 'remediation_owner'],
    "order_view": ['*', 'function_id', 'issue_id', 'username', 'function_info', 'issue_status', 'issue_description', 'issue_type', 'issue_level', 'occurrence_date', 'report_date', 'importance']
    # Add more tables like this:
    # "your_table_name": ['*', 'col1', 'col2', 'col3']
}


# --- New Function to Parse CREATE Statements ---
def generate_table_mapping_from_create_statements(create_statements):
    """
    Parse CREATE TABLE/VIEW statements to generate the TABLE_COLUMN_MAPPING dictionary
    
    Args:
        create_statements (str): A string containing one or more CREATE TABLE/VIEW statements
        
    Returns:
        dict: A dictionary mapping table names to lists of column names
    """
    result_mapping = {}
    
    # Handle multiple statements by splitting on semicolons
    statements = create_statements.split(';')
    
    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue
            
        # Check if it's a CREATE TABLE or CREATE VIEW statement
        create_match = re.match(r'\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?([`"\[\w\.]+)(?:\s+AS\b)?', 
                               statement, re.IGNORECASE)
        
        if not create_match:
            continue
            
        # Extract table/view name and clean it
        table_name = create_match.group(1)
        # Remove schema if present
        if '.' in table_name:
            table_name = table_name.split('.')[-1]
        # Remove quotes/brackets if present
        table_name = re.sub(r'[`"\[\]]', '', table_name).lower()
        
        # Initialize with * for wildcard support
        columns = ['*']
        
        # Check if it's a CREATE VIEW AS SELECT statement
        view_select_match = re.search(r'\bAS\b\s*\(\s*SELECT\s+(.*?)(?:\bFROM\b|\);?|$)', 
                                     statement, re.IGNORECASE | re.DOTALL)
        
        if view_select_match:
            # This is a CREATE VIEW with SELECT - extract columns from SELECT clause
            select_columns = view_select_match.group(1).strip()
            col_list = re.split(r',\s*', select_columns)
            for col in col_list:
                # Handle column aliases (AS keyword)
                alias_match = re.search(r'\bAS\b\s+([`"\[\w]+)', col, re.IGNORECASE)
                if alias_match:
                    col_name = alias_match.group(1)
                else:
                    # Take the last part after any dots, functions, or expressions
                    parts = re.split(r'\.', col)
                    col_name = parts[-1].strip() if parts else col.strip()
                    # Remove any functions or expressions
                    col_name = re.sub(r'.*\(|\).*', '', col_name).strip()
                    
                # Clean up quotes and brackets
                col_name = re.sub(r'[`"\[\]]', '', col_name).lower()
                if col_name and col_name not in columns and col_name != '*':
                    columns.append(col_name)
        else:
            # This is likely a CREATE TABLE statement - extract columns from definition
            # Find the part between the first ( and the last )
            columns_match = re.search(r'\(\s*(.*?)\s*\)[^)]*$', statement, re.DOTALL)
            if columns_match:
                columns_def = columns_match.group(1).strip()
                # Split on commas, but not commas inside parentheses (for handling DEFAULT constraints)
                # This is a simplified approach - complex defaults might need more robust parsing
                col_defs = []
                current_def = ""
                paren_level = 0
                
                for char in columns_def:
                    if char == '(':
                        paren_level += 1
                        current_def += char
                    elif char == ')':
                        paren_level -= 1
                        current_def += char
                    elif char == ',' and paren_level == 0:
                        col_defs.append(current_def.strip())
                        current_def = ""
                    else:
                        current_def += char
                
                if current_def.strip():
                    col_defs.append(current_def.strip())
                
                for col_def in col_defs:
                    # Skip constraints, primary keys, foreign keys, etc.
                    if re.match(r'\s*(CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|INDEX)', 
                               col_def, re.IGNORECASE):
                        continue
                    
                    # Extract column name from definition
                    col_match = re.match(r'\s*([`"\[\w]+)', col_def)
                    if col_match:
                        col_name = col_match.group(1)
                        # Clean up quotes and brackets
                        col_name = re.sub(r'[`"\[\]]', '', col_name).lower()
                        if col_name not in columns:
                            columns.append(col_name)
        
        # Add to our mapping if we found columns
        if len(columns) > 1:  # More than just '*'
            result_mapping[table_name] = columns
    
    return result_mapping


# --- SQLQueryInspector Class (Unchanged) ---
class SQLQueryInspector:
    def __init__(self, query):
        self.query = query
        self.issues = []

    def inspect_query(self):
        # Check for SELECT statements (allowing WITH ... SELECT)
        if not (re.match(r'\s*SELECT', self.query, re.IGNORECASE | re.DOTALL) or
                re.match(r'\s*WITH\s+.*?\s+AS\s*\(.*?\)\s*SELECT', self.query, re.IGNORECASE | re.DOTALL)):
                self.issues.append("Only SELECT statements or CTEs (WITH...SELECT) are allowed.")

        # Check for potential SQL injection / Disallowed Keywords
        disallowed_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        for keyword in disallowed_keywords:
            if re.search(fr'\b{keyword}\b', self.query, re.IGNORECASE):
                self.issues.append(f"Potential disallowed operation detected: '{keyword}'.")

        # Check for unsafe keywords
        unsafe_keywords = ['xp_cmdshell', 'exec(\s|\()', 'sp_', 'xp_', ';\s*--']
        for keyword_pattern in unsafe_keywords:
             if re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE):
                 match = re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE)
                 actual_keyword = match.group(0).strip() if match else keyword_pattern
                 self.issues.append(f"Potentially unsafe SQL pattern '{actual_keyword}' detected.")

        # Check for use of LIMIT/OFFSET without ORDER BY
        if re.search(r'\b(LIMIT|OFFSET)\b', self.query, re.IGNORECASE) and not re.search(r'\bORDER\s+BY\b', self.query, re.IGNORECASE):
            self.issues.append("Use of LIMIT/OFFSET without ORDER BY may result in unpredictable results.")

        # Check for use of semicolons in the middle of the query (allowing at the end)
        if re.search(r';(?!\s*(--.*)?$)', self.query.strip()):
             self.issues.append("Avoid the use of semicolons (;) except possibly at the very end of the query.")

        # Check for use of JOIN without ON clause (Heuristic)
        join_pattern = r'\bJOIN\s+([\w.]+)(\s+\w+)?(?!\s+(ON|USING)\b)'
        potential_cartesian_joins = re.findall(join_pattern, self.query, re.IGNORECASE)
        if potential_cartesian_joins:
             if not re.search(r'\bCROSS\s+JOIN\b', self.query, re.IGNORECASE):
                join_match = re.search(join_pattern, self.query, re.IGNORECASE)
                if join_match:
                    substring_after_join = self.query[join_match.end():]
                    # Check if ON or USING appears *somewhere* after the JOIN match before the next JOIN or end of query section
                    # This is still a heuristic. A full parse is needed for perfect accuracy.
                    next_join_match = re.search(r'\bJOIN\b', substring_after_join, re.IGNORECASE)
                    search_area = substring_after_join if not next_join_match else substring_after_join[:next_join_match.start()]
                    if not re.search(r'\b(ON|USING)\b', search_area, re.IGNORECASE | re.DOTALL):
                        self.issues.append("Use of JOIN without an ON/USING clause may result in a Cartesian product. Specify join conditions or use CROSS JOIN.")


        # Check for use of UNION (Basic check)
        if re.search(r'\bUNION\b', self.query, re.IGNORECASE):
            self.issues.append("UNION queries detected. Ensure column counts and types match in each SELECT.")

        if self.issues:
            issues_str = "Detected issues while validating SQL query:\n" + "\n".join(f"- {issue}" for issue in self.issues)
            return issues_str
        else:
            return self.query


# --- Aggregate Pattern (Global) ---
agg_pattern = re.compile(r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:\*|\w+|\bDISTINCT\b\s+\w+)\s*\)', re.IGNORECASE)

# --- check_and_clean_columns (Unchanged) ---
def check_and_clean_columns(columns_raw, ctes_present, known_base_table_aliases, known_base_table_names):
    cleaned_columns_for_validation = []
    known_prefixes = known_base_table_aliases.union(known_base_table_names)
    for col_raw in columns_raw:
        if agg_pattern.match(col_raw):
            continue
        if ctes_present:
            if '.' in col_raw:
                parts = col_raw.split('.', 1)
                prefix = parts[0].lower()
                col_name = parts[1]
                if prefix in known_prefixes:
                    cleaned_columns_for_validation.append(col_name.lower())
        else:
            if '.' in col_raw:
                col_name = col_raw.split('.')[-1].lower()
                cleaned_columns_for_validation.append(col_name)
            elif col_raw != '*':
                 col_name = col_raw.lower()
                 cleaned_columns_for_validation.append(col_name)
    return list(set(cleaned_columns_for_validation))

# --- validate_columns (Unchanged - Includes user-friendly error message) ---
def validate_columns(extracted_tables, cleaned_columns_for_validation, table_column_mapping):
    extracted_tables_lower = [t.lower() for t in extracted_tables]
    valid_columns_for_query = set(['*'])
    unknown_tables = []
    for table_name_lower in extracted_tables_lower:
        if table_name_lower in table_column_mapping:
            valid_columns_for_query.update(col.lower() for col in table_column_mapping[table_name_lower])
        else:
            if table_name_lower not in unknown_tables:
                 unknown_tables.append(table_name_lower)
    if unknown_tables:
        error_message = f"Query references undefined tables: {', '.join(sorted(unknown_tables))}"
        return False, [error_message]
    invalid_columns = []
    for col in cleaned_columns_for_validation:
        if col not in valid_columns_for_query:
             if col != '*':
                invalid_columns.append(col)
    if invalid_columns:
        sorted_invalid_cols = sorted(list(set(invalid_columns)))
        sorted_tables_referenced = sorted(list(set(extracted_tables_lower)))
        error_message = f"Columns [{', '.join(sorted_invalid_cols)}] are not defined for the referenced tables [{', '.join(sorted_tables_referenced)}]"
        return False, [error_message]
    else:
        return True, []

# --- query_validator (Unchanged, except optional table_column_mapping parameter) ---
def query_validator(query, local_table_column_mapping=None):
    # Use the provided mapping or fall back to the global one
    if local_table_column_mapping is None:
        local_table_column_mapping = TABLE_COLUMN_MAPPING
        
    inspector = SQLQueryInspector(query)
    output_query = inspector.inspect_query()
    if output_query != query:
        return output_query
    else:
        try:
            parser = Parser(query)
            base_tables = [t.lower() for t in parser.tables]
            columns_raw = parser.columns
            ctes_present = bool(parser.with_names)
            base_table_aliases = {alias.lower() for alias, table in parser.tables_aliases.items() if table.lower() in base_tables}
            known_base_table_names = {t.lower() for t in base_tables if t.lower() in local_table_column_mapping}
            if not base_tables and not ctes_present:
                 is_simple_select = True
                 if columns_raw:
                     for c in columns_raw:
                         if not (c.isdigit() or c == '*' or agg_pattern.match(c) or re.match(r'^\w+\(\s*\)$', c)):
                              is_simple_select = False
                              break
                 if is_simple_select:
                     return query
                 else:
                     return "Validation Error: Columns specified without a valid table or CTE reference."
            columns_cleaned_for_validation = check_and_clean_columns(
                columns_raw, ctes_present, base_table_aliases, known_base_table_names
            )
            is_valid, validation_issues = validate_columns(
                base_tables, columns_cleaned_for_validation, local_table_column_mapping
            )
            if is_valid:
                return query
            else:
                return f"Validation Error: {', '.join(validation_issues)}"
        except Exception as e:
             print(f"Error during query parsing or validation: {e}")
             if "Unknown token" in str(e) or "Parse" in str(e):
                 return f"Validation Error: Failed to parse the query structure. Check syntax near error mentioned: ({e})"
             else:
                 return f"Validation Error: An unexpected issue occurred during validation. ({e})"


# --- Direct Test Execution Area ---
if __name__ == "__main__":
    print("--- Starting SQL Query Validation Tests ---")
    
    # Example usage of CREATE statement parser
    create_statements_example = """
    CREATE TABLE ecommerce_product_detail (
        product_id INT PRIMARY KEY,
        count INT,
        user_id VARCHAR(50),
        category VARCHAR(100),
        department VARCHAR(100),
        item_id VARCHAR(50),
        assigned_to VARCHAR(100),
        tags VARCHAR(255),
        priority VARCHAR(20),
        comments TEXT,
        remediation_owner VARCHAR(100)
    );
    
    CREATE TABLE ecommerce_product_metadata (
        product_id INT PRIMARY KEY,
        dataset_definition_id INT,
        query_details TEXT,
        product_name VARCHAR(255),
        sql_filter TEXT,
        product_metadata TEXT,
        category VARCHAR(100),
        comments TEXT,
        priority VARCHAR(20),
        remediation_owner VARCHAR(100)
    );
    
    CREATE VIEW order_view AS 
    SELECT 
        function_id,
        issue_id,
        username,
        function_info,
        issue_status,
        issue_description,
        issue_type,
        issue_level,
        occurrence_date,
        report_date,
        importance
    FROM order_table;
    """
    
    print("\n--- Testing CREATE Statement Parser ---")
    generated_mapping = generate_table_mapping_from_create_statements(create_statements_example)
    print("Generated table mapping:")
    for table, columns in generated_mapping.items():
        print(f"{table}: {columns}")

    # Structure: [Query String, Expected Result ('Pass' or 'Fail')]
    test_cases = [
        # === Valid Cases ===
        ["SELECT product_id, product_name FROM ecommerce_product_metadata WHERE category = 'Electronics'", 'Pass'],
        ["SELECT * FROM order_view WHERE issue_level = 'High'", 'Pass'],
        ["SELECT ov.username, ov.issue_id FROM order_view ov WHERE ov.importance > 5", 'Pass'],
        ["SELECT product_id AS PID, product_name AS Name FROM ecommerce_product_metadata", 'Pass'],
        ["SELECT p.product_id, p.category FROM ecommerce_product_detail p WHERE p.department = 'Home'", 'Pass'],
        ["select product_id from ecommerce_product_metadata;", 'Pass'],
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.category, pc.ProdCount
         FROM ProductCounts pc
         WHERE pc.ProdCount > 10
         """, 'Pass'],
        ["SELECT 1", 'Pass'],
        ["SELECT count(*) from ecommerce_product_detail", 'Pass'],
        ["SELECT COUNT(*) AS total_count FROM ecommerce_product_detail", 'Pass'],
        ["SELECT category, count(product_id) as NumberOfProducts FROM ecommerce_product_detail GROUP BY category", 'Pass'],

        # --- Valid JOIN Cases ---
        ["SELECT o.username, p.product_name FROM order_view o JOIN ecommerce_product_metadata p ON o.username = p.remediation_owner", 'Pass'],
        ["""
         WITH HighIssues AS (
             SELECT issue_id, username AS user_involved
             FROM order_view
             WHERE issue_level = 'High'
         )
         SELECT epm.product_name, hi.user_involved
         FROM ecommerce_product_metadata epm
         JOIN HighIssues hi ON epm.remediation_owner = hi.user_involved
         WHERE epm.priority = 'Critical'
        """, 'Pass'],
        ["SELECT epm.product_name, ov.issue_description FROM ecommerce_product_metadata epm LEFT JOIN order_view ov ON epm.product_id = ov.function_id WHERE epm.category = 'Tools'", 'Pass'], # LEFT JOIN
        ["SELECT epd.item_id, epm.product_name, ov.issue_status FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm ON epd.product_id = epm.product_id JOIN order_view ov ON epm.remediation_owner = ov.username WHERE epd.department = 'Garden'", 'Pass'], # Multiple JOINs
        ["SELECT p1.product_name, p2.product_name FROM ecommerce_product_metadata p1 JOIN ecommerce_product_metadata p2 ON p1.category = p2.category WHERE p1.product_id < p2.product_id", 'Pass'], # Self JOIN
        ["SELECT epm.product_name, ov.username FROM ecommerce_product_metadata epm CROSS JOIN order_view ov WHERE epm.category = 'Books'", 'Pass'], # CROSS JOIN
        ["SELECT epd.item_id, epm.product_name FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm USING (product_id)", 'Pass'], # JOIN USING

        # === Invalid Cases (Inspector Checks) ===
        ["UPDATE ecommerce_product_metadata SET category = 'Outdoor' WHERE product_id = 1", 'Fail'],
        ["DELETE FROM order_view WHERE issue_id < 100", 'Fail'],
        ["SELECT product_id FROM ecommerce_product_metadata LIMIT 5", 'Fail'],
        ["SELECT product_id; SELECT username FROM order_view", 'Fail'],
        ["SELECT * FROM ecommerce_product_detail WHERE product_id = 1; -- xp_cmdshell('dir')", 'Fail'],
        ["SELECT p.product_name, u.username FROM ecommerce_product_metadata p JOIN order_view u", 'Fail'], # JOIN without ON/USING (Inspector check)

        # === Invalid Cases (Column/Table/Parse Checks) ===
        ["SELECT product_id, non_existent_column FROM ecommerce_product_metadata", 'Fail'],
        ["SELECT * FROM non_existent_table", 'Fail'],
        ["SELECT ov.username, ov.bad_column FROM order_view ov", 'Fail'],
        ["SELECT user_id FROM user_data", 'Fail'],
        ["SELECT product_id FROM ecommerce_product_metadata m JOIN order_view o ON m.product_id = o.function_id WHERE o.invalid_col = 1", 'Fail'], # Invalid column in WHERE
        ["SELECT COUNT(*) AS total_count, invalid_column FROM ecommerce_product_detail", 'Fail'],
        ["SELECT FROM table", "Fail"], # Parse error
        ["SELECT col1 FROM ecommerce_product_detail JOIN non_existent_table nx ON nx.id = ecommerce_product_detail.product_id", "Fail"], # Invalid table in JOIN
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.ProdCount, epd.non_existent_base_column
         FROM ProductCounts pc
         JOIN ecommerce_product_detail epd ON pc.category = epd.category
         """, 'Fail'], # Invalid column from base table joined with CTE
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.category, pc.InvalidCTEName
         FROM ProductCounts pc
         """, 'Pass'], # Limitation: Invalid column from CTE itself is not checked

        # --- Invalid JOIN Cases (Column/Table Validation) ---
        ["SELECT epm.product_name FROM ecommerce_product_metadata epm JOIN order_view ov ON epm.product_id = ov.non_existent_key", 'Fail'], # Invalid column in JOIN ON clause
        ["SELECT epm.product_name, ov.invalid_column_from_join FROM ecommerce_product_metadata epm JOIN order_view ov ON epm.remediation_owner = ov.username", 'Fail'], # Invalid column in SELECT from joined table
        ["SELECT p1.product_name, p2.invalid_self_col FROM ecommerce_product_metadata p1 JOIN ecommerce_product_metadata p2 ON p1.category = p2.category WHERE p1.product_id < p2.product_id", 'Fail'], # Invalid column in Self JOIN SELECT
        ["SELECT epd.item_id, epm.product_name, ov.invalid_multi_join_col FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm ON epd.product_id = epm.product_id JOIN order_view ov ON epm.remediation_owner = ov.username WHERE epd.department = 'Garden'", 'Fail'], # Invalid column from 3rd table in multi-join

    ]

    # Run tests with generated mapping from CREATE statements
    definitions_to_use = generated_mapping  # Use the mapping generated from CREATE statements
    passed_count = 0
    failed_count = 0

    print("\n--- Running Tests with Generated Mapping ---")
    for i, (query, expected_result) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Query  : {query.strip()}")
        print(f"Expect : {expected_result}")

        actual_result_msg = query_validator(query, definitions_to_use)

        is_pass = True
        if "Detected issues" in actual_result_msg or "Validation Error" in actual_result_msg:
            is_pass = False

        actual_outcome = 'Pass' if is_pass else 'Fail'
        print(f"Actual : {actual_outcome}")

        if actual_outcome == expected_result:
            print("Result : CORRECT")
            passed_count += 1
        else:
            print(f"Result : INCORRECT ******")
            failed_count += 1

        if not is_pass:
            print(f"Reason : {actual_result_msg}")

    print("\n--- Validation Tests Summary ---")
    print(f"Total Cases: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("--- Testing Complete ---")


# --- Example function to demonstrate the overall usage flow ---
def validate_sql_from_create_statements(create_statements, query_to_validate):
    """
    End-to-end function to validate SQL queries based on table definitions from CREATE statements
    
    Args:
        create_statements (str): SQL CREATE TABLE/VIEW statements
        query_to_validate (str): SQL query to validate
        
    Returns:
        str: Validation result (either the original query if valid, or error message)
    """
    # Generate table mapping from CREATE statements
    mapping = generate_table_mapping_from_create_statements(create_statements)
    
    # If no mapping could be generated, use the default mapping
    if not mapping:
        mapping = TABLE_COLUMN_MAPPING
        print("Warning: No valid table definitions found in CREATE statements. Using default mapping.")
    
    # Validate the query using the generated mapping
    return query_validator(query_to_validate, mapping)
