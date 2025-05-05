import re
import logging # Logging setup remains commented as in original code
from sql_metadata import Parser


# --- Centralized Configuration Section ---
# EDIT THIS DICTIONARY TO CHANGE TABLES AND COLUMNS
# Keys = Table names (as expected from sql-metadata parser, often lowercase)
# Values = List of allowed column names (use lowercase for consistency)
TABLE_COLUMN_MAPPING = {
    "ecommerce_product_detail": ['*', 'product_id', 'count', 'user_id', 'category', 'department', 'item_id', 'assigned_to', 'tags', 'priority', 'comments', 'remediation_owner'],
    "ecommerce_product_metadata": ['*', 'product_id', 'dataset_definition_id', 'query_details', 'product_name', 'sql_filter', 'product_metadata', 'category', 'comments', 'priority', 'remediation_owner'],
    "order_view": ['*', 'function_id', 'issue_id', 'username', 'function_info', 'issue_status', 'issue_description', 'issue_type', 'issue_level', 'occurrence_date', 'report_date', 'importance']
    # Add more tables like this:
    # "your_table_name": ['*', 'col1', 'col2', 'col3']
}
# --- End Configuration Section ---


# --- Original Code Structure (with modifications for centralized mapping & alias handling) ---

class SQLQueryInspector:
    # No changes to this class - kept exactly as in the previous version
    def __init__(self, query):
        self.query = query
        # self.logger = self._setup_logger() # Logger setup kept commented
        self.issues = []

    # def _setup_logger(self): # Logger setup kept commented
    #     # ... (original logger setup) ...
    #     return logger

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
        if re.search(r'\bJOIN\s+([\w.]+)(\s+\w+)?(?!\s+(ON|USING)\b)', self.query, re.IGNORECASE):
             if not re.search(r'\bCROSS\s+JOIN\b', self.query, re.IGNORECASE):
                self.issues.append("Use of JOIN without an ON/USING clause may result in a Cartesian product. Specify join conditions or use CROSS JOIN.")

        # Check for use of UNION (Basic check)
        if re.search(r'\bUNION\b', self.query, re.IGNORECASE):
            self.issues.append("UNION queries detected. Ensure column counts and types match in each SELECT.")

        if self.issues:
            issues_str = "Detected issues while validating SQL query:\n" + "\n".join(f"- {issue}" for issue in self.issues)
            return issues_str
        else:
            return self.query

# --- Modified check_and_clean_columns ---
def check_and_clean_columns(columns_raw):
    """
    Cleans column names: removes prefixes, lowercases.
    **Modification**: Attempts to filter out aliases of common aggregate functions
    so they are not validated against table schemas.
    Accepts the raw list of columns from the parser.
    """
    cleaned_columns_for_validation = []
    # Regex to detect common aggregate functions (case-insensitive) potentially followed by 'AS alias'
    # This pattern tries to match the function structure itself, not just the alias.
    # Matches: AGG(col), AGG(*), AGG(distinct col) potentially followed by AS alias
    agg_pattern = re.compile(r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:\*|\w+|\bDISTINCT\b\s+\w+)\s*\)', re.IGNORECASE)

    for col_raw in columns_raw:
        # Check if the raw column string starts with an aggregate pattern
        if agg_pattern.match(col_raw):
            # If it's an aggregate function, we assume it's syntactically valid if the inspector passed.
            # We don't add it (or its alias if present) to the list of columns
            # that need to be explicitly defined in the TABLE_COLUMN_MAPPING.
            # print(f"Debug: Skipping aggregate column/alias from validation list: {col_raw}") # Optional debug
            continue # Skip this item

        # If not an aggregate, process as before for validation list
        col = col_raw # Use the raw name for processing
        if '.' in col:
            # Take only the part after the last dot, lowercase
            cleaned_columns_for_validation.append(col.split('.')[-1].lower())
        else:
             # Exclude '*' and lowercase
            if col != '*':
                 cleaned_columns_for_validation.append(col.lower())

    # Return unique column names intended for validation against schema definitions
    return list(set(cleaned_columns_for_validation))
# --- End Modified check_and_clean_columns ---


def validate_columns(extracted_tables, cleaned_columns_for_validation, table_column_mapping):
    """
    Validates if cleaned columns (excluding aggregates/aliases) belong to the
    allowed columns for the extracted tables based on the mapping.
    """
    extracted_tables_lower = [t.lower() for t in extracted_tables]
    valid_columns_for_query = set(['*']) # Base set of allowed columns
    unknown_tables = []

    for table_name_lower in extracted_tables_lower:
        if table_name_lower in table_column_mapping:
            valid_columns_for_query.update(col.lower() for col in table_column_mapping[table_name_lower])
        else:
            unknown_tables.append(table_name_lower)
            print(f"Warning: Table '{table_name_lower}' not found in definitions.")

    if unknown_tables:
        # Strict option: return False, [f"Query references undefined tables: {', '.join(unknown_tables)}"]
        pass # Continue validation based on known tables

    # Check the cleaned, extracted columns (which should NOT include agg aliases)
    invalid_columns = []
    for col in cleaned_columns_for_validation: # Use the list filtered by check_and_clean_columns
        if col not in valid_columns_for_query:
             # This check is less critical now as aggregates are pre-filtered, but keep as fallback
             if not re.match(r'^(count|sum|avg|min|max)\(.*\)$', col, re.IGNORECASE):
                 invalid_columns.append(col)


    if invalid_columns:
        return False, list(invalid_columns)
    else:
        return True, []


def query_validator(query, local_table_column_mapping):
    """
    Main validator function using the centralized mapping.
    Passes raw columns to check_and_clean_columns.
    """
    inspector = SQLQueryInspector(query)
    output_query = inspector.inspect_query()

    if output_query != query:
        return output_query # Return the error message from inspector

    else:
        try:
            parser = Parser(query)
            tables = [t.lower() for t in parser.tables]
            columns_raw = parser.columns # Get raw columns from parser

            # Handle cases like "SELECT 1" or "SELECT func()"
            if not tables:
                 if not columns_raw or all(agg_pattern.match(c) or c.isdigit() for c in columns_raw): # Check using the agg pattern
                     # print("Query has no tables but seems valid (constant/function/agg). Passing.") # Debug print
                     return query
                 else:
                      # print(f"Debug: No tables, but columns found: {columns_raw}") # Debug print
                      # Check if any non-function-like columns exist without a table
                      if any(not agg_pattern.match(c) and not c.isdigit() for c in columns_raw):
                          return "Validation Error: Columns specified without a valid table reference."
                      else: # Only functions/constants found
                          return query


            # Pass RAW columns to cleaner function
            columns_cleaned_for_validation = check_and_clean_columns(columns_raw)
            # print(f"Debug: Columns requiring validation: {columns_cleaned_for_validation}") # Optional debug print

            if tables: # Only validate columns if tables were found
                # Validate the CLEANED list (which excludes agg aliases)
                is_valid, invalid_cols = validate_columns(tables, columns_cleaned_for_validation, local_table_column_mapping)

                if is_valid:
                    return query # Return original query if valid
                else:
                    return f"Validation Error: Invalid columns detected: {', '.join(invalid_cols)}"
            # This part might be redundant given the 'not tables' check above, but keep for safety
            elif not columns_cleaned_for_validation:
                 if 'FROM' in query.upper() and not tables:
                      return "Validation Error: Could not identify tables referenced in the query."
                 else:
                      return query
            else: # Tables list is empty, but cleaned_columns_for_validation is not (shouldn't happen with current logic)
                 return "Validation Error: Columns found but associated table could not be determined."


        except Exception as e:
             print(f"Error during query parsing: {e}")
             return f"Validation Error: Failed to parse the query structure. ({e})"


# --- Direct Test Execution Area ---
if __name__ == "__main__":
    print("--- Starting SQL Query Validation Tests ---")

    # Structure: [Query String, Expected Result ('Pass' or 'Fail')]
    test_cases = [
        # === Valid Cases ===
        ["SELECT product_id, product_name FROM ecommerce_product_metadata WHERE category = 'Electronics'", 'Pass'],
        ["SELECT * FROM order_view WHERE issue_level = 'High'", 'Pass'],
        ["SELECT ov.username, ov.issue_id FROM order_view ov WHERE ov.importance > 5", 'Pass'],
        ["SELECT product_id AS PID, product_name AS Name FROM ecommerce_product_metadata", 'Pass'], # Column alias OK
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
         """, 'Pass'], # WITH clause should pass inspector and columns 'category', 'ProdCount' are handled (ProdCount ignored by cleaner)
        ["SELECT o.username, p.product_name FROM order_view o JOIN ecommerce_product_metadata p ON o.username = p.remediation_owner", 'Pass'],
        ["SELECT 1", 'Pass'],
        ["SELECT count(*) from ecommerce_product_detail", 'Pass'], # Aggregate without alias
        ["SELECT COUNT(*) AS total_count FROM ecommerce_product_detail", 'Pass'], # *** Aggregate WITH alias (Edge Case Test) ***
        ["SELECT category, count(product_id) as NumberOfProducts FROM ecommerce_product_detail GROUP BY category", 'Pass'], # Mixed cols + aggregate alias

        # === Invalid Cases (Inspector Checks) ===
        ["UPDATE ecommerce_product_metadata SET category = 'Outdoor' WHERE product_id = 1", 'Fail'],
        ["DELETE FROM order_view WHERE issue_id < 100", 'Fail'],
        ["SELECT product_id FROM ecommerce_product_metadata LIMIT 5", 'Fail'],
        ["SELECT product_id; SELECT username FROM order_view", 'Fail'],
        ["SELECT * FROM ecommerce_product_detail WHERE product_id = 1; -- xp_cmdshell('dir')", 'Fail'], # Potential unsafe keyword in comment still caught by simple regex
        # ["SELECT p.product_name, u.username FROM ecommerce_product_metadata p JOIN non_existent_user_table u", 'Fail'], # JOIN without ON (covered below better)
        ["SELECT p.product_name, u.username FROM ecommerce_product_metadata p JOIN order_view u", 'Fail'], # JOIN without ON/USING


        # === Invalid Cases (Column/Table/Parse Checks) ===
        ["SELECT product_id, non_existent_column FROM ecommerce_product_metadata", 'Fail'],
        ["SELECT * FROM non_existent_table", 'Fail'], # Should fail as table not found by parser/validator
        ["SELECT ov.username, ov.bad_column FROM order_view ov", 'Fail'],
        ["SELECT user_id FROM user_data", 'Fail'], # Fails table check (user_data not in mapping)
        ["SELECT product_id FROM ecommerce_product_metadata m JOIN order_view o ON m.product_id = o.function_id WHERE o.invalid_col = 1", 'Fail'],
        ["SELECT COUNT(*) AS total_count, invalid_column FROM ecommerce_product_detail", 'Fail'], # Mix valid agg alias and invalid column
        ["SELECT FROM table", "Fail"], # Invalid syntax, parser error expected
    ]

    definitions_to_use = TABLE_COLUMN_MAPPING
    passed_count = 0
    failed_count = 0

    for i, (query, expected_result) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Query  : {query.strip()}")
        print(f"Expect : {expected_result}")

        actual_result_msg = query_validator(query, definitions_to_use)

        # Determine actual outcome based on validator's return message
        is_pass = True
        if "Detected issues" in actual_result_msg or "Validation Error" in actual_result_msg or "Error during query parsing" in actual_result_msg :
             is_pass = False

        actual_outcome = 'Pass' if is_pass else 'Fail'
        print(f"Actual : {actual_outcome}")

        if actual_outcome == expected_result:
            print("Result : CORRECT")
            passed_count += 1
        else:
            print(f"Result : INCORRECT ******")
            failed_count += 1

        # If failed, print the reason
        if not is_pass:
            print(f"Reason : {actual_result_msg}")

    print("\n--- Validation Tests Summary ---")
    print(f"Total Cases: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("--- Testing Complete ---")
