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


# --- SQLQueryInspector Class (Unchanged) ---
class SQLQueryInspector:
    # ... (Keep the class exactly as it was in the previous version) ...
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
                # Basic check: if 'ON' or 'USING' appears *after* the JOIN keyword in the remaining string, assume okay.
                # This is a weak heuristic.
                join_match = re.search(join_pattern, self.query, re.IGNORECASE)
                if join_match:
                    substring_after_join = self.query[join_match.end():]
                    if not re.search(r'\b(ON|USING)\b', substring_after_join, re.IGNORECASE | re.DOTALL):
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

# --- MODIFIED check_and_clean_columns ---
def check_and_clean_columns(columns_raw, ctes_present, known_base_table_aliases, known_base_table_names):
    """
    Cleans column names for validation.
    **Modification**: If CTEs are present, only adds columns qualified with known
                      base table names/aliases to the validation list. Skips
                      unqualified columns or those qualified with unknown prefixes
                      (assumed CTE aliases). Also skips aggregate function calls.
    """
    cleaned_columns_for_validation = []
    # Combine known base table names and their aliases for easier checking
    known_prefixes = known_base_table_aliases.union(known_base_table_names)
    # print(f"Debug Clean: Known prefixes for validation: {known_prefixes}") # Debug

    for col_raw in columns_raw:
        # print(f"Debug Clean: Processing raw column: {col_raw}") # Debug
        # Skip aggregate function calls like COUNT(*), SUM(col) etc.
        if agg_pattern.match(col_raw):
            # print(f"Debug Clean: Skipping aggregate: {col_raw}") # Debug
            continue

        # If CTEs are present, apply stricter filtering for validation list
        if ctes_present:
            if '.' in col_raw:
                # Split only on the first dot if present
                parts = col_raw.split('.', 1)
                prefix = parts[0].lower()
                col_name = parts[1] # Keep original case temporarily if needed, lowercase later

                # ONLY add for validation if the prefix matches a KNOWN BASE TABLE name or alias
                if prefix in known_prefixes:
                    # print(f"Debug Clean: CTE present, known prefix '{prefix}'. Adding '{col_name.lower()}' for validation.") # Debug
                    cleaned_columns_for_validation.append(col_name.lower())
                # else: # Prefix is unknown (likely CTE alias) - DO NOT add for validation
                    # print(f"Debug Clean: CTE present, unknown prefix '{prefix}'. Skipping '{col_raw}'.") # Debug

            # else: # No prefix and CTEs are present. Assume it might be from CTE. DO NOT validate.
                # print(f"Debug Clean: CTE present, no prefix. Skipping '{col_raw}'.") # Debug
                pass # Explicitly do nothing

        # No CTEs present, process as before (validate almost everything)
        else:
            if '.' in col_raw:
                # Take only the part after the last dot, lowercase
                col_name = col_raw.split('.')[-1].lower()
                # print(f"Debug Clean: No CTEs, dot found. Adding '{col_name}' for validation.") # Debug
                cleaned_columns_for_validation.append(col_name)
            elif col_raw != '*':
                 col_name = col_raw.lower()
                 # print(f"Debug Clean: No CTEs, no dot. Adding '{col_name}' for validation.") # Debug
                 cleaned_columns_for_validation.append(col_name)

    # Return unique column names intended for validation against base table schema definitions
    unique_cols = list(set(cleaned_columns_for_validation))
    # print(f"Debug Clean: Final columns for validation: {unique_cols}") # Debug
    return unique_cols
# --- End Modified check_and_clean_columns ---


# --- validate_columns (Unchanged from previous version with strict table check) ---
def validate_columns(extracted_tables, cleaned_columns_for_validation, table_column_mapping):
    """
    Validates if cleaned columns belong to the allowed columns for the extracted BASE tables.
    Fails validation immediately if any referenced BASE table is not found in the mapping.
    """
    # print(f"Debug Validate: Validating tables: {extracted_tables}") # Debug
    # print(f"Debug Validate: Validating columns: {cleaned_columns_for_validation}") # Debug
    extracted_tables_lower = [t.lower() for t in extracted_tables]
    valid_columns_for_query = set(['*']) # Base set of allowed columns
    unknown_tables = [] # List to store tables not found in the mapping

    # --- Check for unknown BASE tables FIRST ---
    for table_name_lower in extracted_tables_lower:
        if table_name_lower in table_column_mapping:
            valid_columns_for_query.update(col.lower() for col in table_column_mapping[table_name_lower])
        else:
            if table_name_lower not in unknown_tables:
                 unknown_tables.append(table_name_lower)

    if unknown_tables:
        error_message = f"Query references undefined tables: {', '.join(sorted(unknown_tables))}"
        # print(f"Debug Validate: Failing due to unknown tables: {unknown_tables}") # Debug
        return False, [error_message]
    # --- End BASE table check ---

    # If all BASE tables were found, proceed to validate the FILTERED columns.
    # print(f"Debug Validate: All base tables known. Allowed columns from base tables: {valid_columns_for_query}") # Debug
    invalid_columns = []
    for col in cleaned_columns_for_validation: # Use the potentially smaller list filtered by check_and_clean_columns
        if col not in valid_columns_for_query:
            # Only flag if it's not '*' (already handled) and not an aggregate (already handled)
            if col != '*':
                invalid_columns.append(col)

    if invalid_columns:
        # print(f"Debug Validate: Failing due to invalid columns: {invalid_columns}") # Debug
        return False, list(invalid_columns) # Return False and the list of invalid columns
    else:
        # print(f"Debug Validate: Passing column validation.") # Debug
        return True, [] # Return True if all columns are valid
# --- End validate_columns ---


# --- MODIFIED query_validator ---
def query_validator(query, local_table_column_mapping):
    """
    Main validator function using the centralized mapping.
    Passes raw columns to check_and_clean_columns along with CTE context.
    """
    inspector = SQLQueryInspector(query)
    output_query = inspector.inspect_query()

    if output_query != query:
        return output_query # Return the error message from inspector

    else:
        try:
            # print(f"Debug QueryValidator: Parsing query: {query[:100]}...") # Debug
            parser = Parser(query)
            # Ensure table names are lowercased immediately for consistent checks
            # parser.tables extracts BASE tables referenced
            base_tables = [t.lower() for t in parser.tables]
            columns_raw = parser.columns # Get raw columns from parser (includes prefixes)
            ctes_present = bool(parser.with_names) # Check if WITH clause exists
            # Get aliases used for base tables (lowercase keys and values)
            base_table_aliases = {alias.lower() for alias, table in parser.tables_aliases.items() if table.lower() in base_tables}
            # Get the names of the base tables defined in our mapping
            known_base_table_names = {t.lower() for t in base_tables if t.lower() in local_table_column_mapping}

            # print(f"Debug QueryValidator: Base Tables: {base_tables}") # Debug
            # print(f"Debug QueryValidator: Raw Columns: {columns_raw}") # Debug
            # print(f"Debug QueryValidator: CTEs Present: {ctes_present}") # Debug
            # print(f"Debug QueryValidator: Base Table Aliases: {base_table_aliases}") # Debug
            # print(f"Debug QueryValidator: Known Base Table Names: {known_base_table_names}") # Debug


            # Handle cases like "SELECT 1" or "SELECT func()" which have no tables
            if not base_tables and not ctes_present: # Modified to check CTEs too
                 is_simple_select = True
                 if columns_raw:
                     for c in columns_raw:
                         if not (c.isdigit() or c == '*' or agg_pattern.match(c) or re.match(r'^\w+\(\s*\)$', c)):
                              is_simple_select = False
                              break
                 if is_simple_select:
                     # print("Debug QueryValidator: Passing simple SELECT without tables/CTEs.") # Debug
                     return query
                 else:
                     # print("Debug QueryValidator: Failing complex SELECT without tables/CTEs.") # Debug
                     return "Validation Error: Columns specified without a valid table or CTE reference."


            # If tables or CTEs ARE present, proceed with validation
            # Pass context to the cleaner function
            columns_cleaned_for_validation = check_and_clean_columns(
                columns_raw,
                ctes_present,
                base_table_aliases,
                known_base_table_names
            )

            # Validate BASE tables first, then the potentially reduced list of columns
            # We only pass BASE tables to validate_columns for checking against the mapping
            is_valid, validation_issues = validate_columns(
                base_tables, # Only check existence of base tables in mapping
                columns_cleaned_for_validation, # Validate the filtered list of columns
                local_table_column_mapping
            )

            if is_valid:
                # print("Debug QueryValidator: Validation Passed.") # Debug
                return query # Return original query if valid
            else:
                # Join the issues list (unknown base tables or invalid columns from base tables)
                error_msg = f"Validation Error: {', '.join(validation_issues)}"
                # print(f"Debug QueryValidator: Validation Failed: {error_msg}") # Debug
                return error_msg

        except Exception as e:
             print(f"Error during query parsing or validation: {e}")
             if "Unknown token" in str(e) or "Parse" in str(e):
                 return f"Validation Error: Failed to parse the query structure. Check syntax near error mentioned: ({e})"
             else:
                 return f"Validation Error: An unexpected issue occurred during validation. ({e})"

# --- End MODIFIED query_validator ---


# --- Direct Test Execution Area ---
if __name__ == "__main__":
    print("--- Starting SQL Query Validation Tests ---")

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
             FROM ecommerce_product_detail -- BASE TABLE
             GROUP BY category
         )
         SELECT pc.category, pc.ProdCount -- Selecting from CTE alias 'pc'
         FROM ProductCounts pc -- Referencing CTE
         WHERE pc.ProdCount > 10
         """, 'Pass'], # <<< EXPECT PASS: 'ProdCount' comes from CTE, validation relaxed.
        ["SELECT o.username, p.product_name FROM order_view o JOIN ecommerce_product_metadata p ON o.username = p.remediation_owner", 'Pass'],
        ["SELECT 1", 'Pass'],
        ["SELECT count(*) from ecommerce_product_detail", 'Pass'],
        ["SELECT COUNT(*) AS total_count FROM ecommerce_product_detail", 'Pass'],
        ["SELECT category, count(product_id) as NumberOfProducts FROM ecommerce_product_detail GROUP BY category", 'Pass'],
        # New test: CTE + Join + Selecting from both Base and CTE
        ["""
         WITH HighIssues AS (
             SELECT issue_id, username AS user_involved
             FROM order_view
             WHERE issue_level = 'High'
         )
         SELECT
            epm.product_name,
            hi.user_involved -- Column from CTE
         FROM ecommerce_product_metadata epm
         JOIN HighIssues hi ON epm.remediation_owner = hi.user_involved
         WHERE epm.priority = 'Critical'
        """, 'Pass'], # <<< EXPECT PASS: 'user_involved' from CTE okay, 'product_name'/'remediation_owner'/'priority' checked against base table.

        # === Invalid Cases (Inspector Checks) ===
        ["UPDATE ecommerce_product_metadata SET category = 'Outdoor' WHERE product_id = 1", 'Fail'],
        ["DELETE FROM order_view WHERE issue_id < 100", 'Fail'],
        ["SELECT product_id FROM ecommerce_product_metadata LIMIT 5", 'Fail'],
        ["SELECT product_id; SELECT username FROM order_view", 'Fail'],
        ["SELECT * FROM ecommerce_product_detail WHERE product_id = 1; -- xp_cmdshell('dir')", 'Fail'],
        ["SELECT p.product_name, u.username FROM ecommerce_product_metadata p JOIN order_view u", 'Fail'],

        # === Invalid Cases (Column/Table/Parse Checks) ===
        ["SELECT product_id, non_existent_column FROM ecommerce_product_metadata", 'Fail'], # Invalid column from base table
        ["SELECT * FROM non_existent_table", 'Fail'], # Invalid base table
        ["SELECT ov.username, ov.bad_column FROM order_view ov", 'Fail'], # Invalid column (via alias) from base table
        ["SELECT user_id FROM user_data", 'Fail'], # Invalid base table
        ["SELECT product_id FROM ecommerce_product_metadata m JOIN order_view o ON m.product_id = o.function_id WHERE o.invalid_col = 1", 'Fail'], # Invalid column (via alias) in WHERE
        ["SELECT COUNT(*) AS total_count, invalid_column FROM ecommerce_product_detail", 'Fail'], # Invalid column alongside aggregate
        ["SELECT FROM table", "Fail"], # Parse error
        ["SELECT col1 FROM ecommerce_product_detail JOIN non_existent_table nx ON nx.id = ecommerce_product_detail.product_id", "Fail"], # Invalid base table in JOIN
        # New test: CTE + Invalid column selected from BASE table
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT
             pc.ProdCount, -- This is OK (from CTE)
             epd.non_existent_base_column -- This is INVALID (from base table)
         FROM ProductCounts pc
         JOIN ecommerce_product_detail epd ON pc.category = epd.category
         """, 'Fail'], # <<< EXPECT FAIL: Tries to select invalid column from base table 'epd'.
        # New test: CTE + Invalid column selected FROM CTE alias itself
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.category, pc.InvalidCTEName -- Selecting non-existent column from CTE alias
         FROM ProductCounts pc
         """, 'Pass'], # <<< EXPECT PASS (Limitation): The current logic doesn't validate columns selected FROM CTEs. It only validates columns selected from base tables.

    ]

    definitions_to_use = TABLE_COLUMN_MAPPING
    passed_count = 0
    failed_count = 0

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
