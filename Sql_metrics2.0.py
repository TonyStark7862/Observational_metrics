import re
import logging # Logging setup remains commented as in original code
from sql_metadata import Parser


# --- DDL Parser for Table/Column Mapping ---
def generate_table_column_mapping_from_ddl(ddl_scripts: str) -> dict:
    """
    Parses DDL CREATE TABLE statements and returns a mapping of table names
    to a list of their column names.
    Includes '*' for all tables.
    """
    mapping = {}
    if not ddl_scripts or not ddl_scripts.strip():
        print("Warning: Empty DDL script provided.")
        return mapping

    try:
        # Using an alias for clarity if Parser is used differently elsewhere,
        # but it's the same Parser class.
        ddl_parser = Parser(ddl_scripts)
    except Exception as e:
        # This might catch very malformed DDL that sql-metadata cannot handle at all
        print(f"Critical Error parsing DDL scripts: {e}")
        return mapping # Return empty mapping on DDL parse failure

    # Expected structure from sql-metadata for CREATE TABLE:
    # parser.columns_definitions = {'table_name': [{'name': 'col1', ...}, {'name': 'col2', ...}]}
    if hasattr(ddl_parser, 'columns_definitions') and ddl_parser.columns_definitions:
        for table_name, col_defs_list in ddl_parser.columns_definitions.items():
            table_name_lower = table_name.lower()
            columns = ['*'] # Always allow '*' for every table
            if isinstance(col_defs_list, list): # Ensure it's a list of column definitions
                for col_def in col_defs_list:
                    if isinstance(col_def, dict) and 'name' in col_def:
                        columns.append(col_def['name'].lower())
                    # sql-metadata might sometimes have simple string list for very simple DDLs
                    # or different structures for complex DDL parts, but 'name' in dict is typical.
            mapping[table_name_lower] = sorted(list(set(columns)))
    
    # Ensure all tables explicitly mentioned in CREATE TABLE (and parsed by .tables) are in the mapping.
    # This handles cases where columns_definitions might miss something or if a table is created
    # with no columns (unlikely but a safeguard).
    for table_name_raw in ddl_parser.tables:
        table_name_l = table_name_raw.lower()
        if table_name_l not in mapping:
            # This could happen if a CREATE TABLE statement was parsed for its name
            # but its columns weren't detailed in columns_definitions for some reason.
            print(f"Info: Table '{table_name_l}' detected by parser but not in columns_definitions. Adding with '*' only.")
            mapping[table_name_l] = ['*']
            
    if not mapping and ddl_parser.tables:
        print("Warning: DDL parser found tables but could not extract column definitions into 'columns_definitions'. Tables will only have '*' as a known column.")
        for table_name_raw in ddl_parser.tables:
            mapping[table_name_raw.lower()] = ['*']

    if not mapping and ddl_scripts.strip():
        print(f"Warning: Could not parse any table/column definitions from the provided DDL.")


    return mapping


# --- SQLQueryInspector Class (Modified) ---
class SQLQueryInspector:
    def __init__(self, query):
        self.query = query
        self.issues = []

    def inspect_query(self):
        # Check for SELECT statements (allowing WITH ... SELECT)
        # This is kept because subsequent column validation is typically for SELECTs.
        # However, DML/injection checks apply regardless.
        is_select_oriented = bool(
            re.match(r'\s*SELECT', self.query, re.IGNORECASE | re.DOTALL) or
            re.match(r'\s*WITH\s+.*?\s+AS\s*\(.*?\)\s*SELECT', self.query, re.IGNORECASE | re.DOTALL)
        )

        if not is_select_oriented:
            # This is more of a note for the validator's typical flow.
            # DML checks below are still crucial.
            # For non-SELECT queries, column validation might not apply or make sense.
            pass # No longer adding an issue here, DML checks are the priority.

        # Check for potential SQL injection / Disallowed Keywords
        # These are critical checks.
        disallowed_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        for keyword in disallowed_keywords:
            # Using word boundaries to avoid matching substrings within other words
            if re.search(fr'\b{keyword}\b', self.query, re.IGNORECASE):
                self.issues.append(f"Potential disallowed operation detected: '{keyword}'.")

        # Check for unsafe keywords/patterns (more specific injection vectors)
        unsafe_keywords_patterns = [
            r'xp_cmdshell',                # SQL Server extended stored procedure
            r'exec\s*\(',                 # EXECUTE string or procedure
            r'\bsp_',                      # Common prefix for system stored procedures
            r'\bxp_',                      # Common prefix for extended stored procedures
            r"['\";]\s*(OR|AND)\s+.*?=.*?" # Basic SQL injection tautology like ' OR '1'='1
        ]
        for keyword_pattern in unsafe_keywords_patterns:
            if re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE):
                match = re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE)
                actual_keyword_found = match.group(0).strip() if match else keyword_pattern
                self.issues.append(f"Potentially unsafe SQL pattern '{actual_keyword_found}' detected.")
        
        # Removed checks:
        # - LIMIT/OFFSET without ORDER BY
        # - Semicolons in the middle of the query
        # - JOIN without ON clause (heuristic)
        # - UNION (Basic check)

        if self.issues:
            issues_str = "Detected issues while validating SQL query:\n" + "\n".join(f"- {issue}" for issue in self.issues)
            return issues_str
        else:
            return self.query


# --- Aggregate Pattern (Global - Unchanged) ---
agg_pattern = re.compile(r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:\*|\w+|\bDISTINCT\b\s+\w+)\s*\)', re.IGNORECASE)

# --- check_and_clean_columns (Unchanged) ---
def check_and_clean_columns(columns_raw, ctes_present, known_base_table_aliases, known_base_table_names):
    cleaned_columns_for_validation = []
    known_prefixes = known_base_table_aliases.union(known_base_table_names)
    for col_raw in columns_raw:
        if agg_pattern.match(col_raw):
            continue
        if ctes_present:
            # If CTEs are present, we only validate columns that are explicitly prefixed
            # with a known base table alias or base table name.
            # Columns derived purely within a CTE and then selected are not validated here
            # against the base TABLE_COLUMN_MAPPING.
            if '.' in col_raw:
                parts = col_raw.split('.', 1)
                prefix = parts[0].lower()
                col_name = parts[1]
                if prefix in known_prefixes: # Check if prefix is a known base table/alias
                    cleaned_columns_for_validation.append(col_name.lower())
                # If prefix is not a known base table/alias, it might be a CTE alias.
                # We are not validating columns from CTEs against the base mapping.
            # else:
                # If column in CTE query has no prefix, it's assumed to be from the CTE itself
                # or a derived column, not directly validated against base tables here.
                # This remains a known limitation: we don't validate the "internals" of CTEs against base schema this way.
        else: # No CTEs present
            if '.' in col_raw:
                col_name = col_raw.split('.')[-1].lower() # Take the column part
                cleaned_columns_for_validation.append(col_name)
            elif col_raw != '*': # Don't add '*' to columns to be validated individually
                col_name = col_raw.lower()
                cleaned_columns_for_validation.append(col_name)
    return list(set(cleaned_columns_for_validation))

# --- validate_columns (Unchanged - Includes user-friendly error message) ---
def validate_columns(extracted_tables, cleaned_columns_for_validation, table_column_mapping):
    extracted_tables_lower = [t.lower() for t in extracted_tables if t] # Ensure t is not None
    valid_columns_for_query = set(['*']) # '*' is always implicitly valid if tables are known
    unknown_tables = []
    
    if not extracted_tables_lower and cleaned_columns_for_validation:
        # This case might arise if columns are selected without a FROM clause (e.g. SELECT 1, 'foo')
        # but somehow `cleaned_columns_for_validation` got populated.
        # The "simple select" check in query_validator should handle most legit cases.
        # If we reach here with columns but no tables, and it's not a simple select, it's an issue.
        is_potentially_simple_select_col = all(
            agg_pattern.match(c) or c.isdigit() or re.match(r"'.*?'", c) or re.match(r"\d+\.\d+", c) or c == '*'
            for c in cleaned_columns_for_validation
        )
        if not is_potentially_simple_select_col:
             return False, ["Columns specified without any valid table reference, and not a simple constant/function select."]


    for table_name_lower in extracted_tables_lower:
        if table_name_lower in table_column_mapping:
            valid_columns_for_query.update(col.lower() for col in table_column_mapping[table_name_lower])
        else:
            if table_name_lower not in unknown_tables: # Avoid duplicates
                unknown_tables.append(table_name_lower)
    
    if unknown_tables:
        error_message = f"Query references undefined tables (not found in DDL): {', '.join(sorted(unknown_tables))}"
        return False, [error_message]

    # If no tables were extracted, but we didn't hit the unknown_tables error,
    # it means it might be a query like "SELECT 1" which is fine, or an error.
    # If cleaned_columns_for_validation is empty at this point, it's likely fine (e.g. SELECT 1 already returned)
    if not extracted_tables_lower and not cleaned_columns_for_validation:
        return True, []


    invalid_columns = []
    for col in cleaned_columns_for_validation:
        if col not in valid_columns_for_query:
            # '*' itself should not be in cleaned_columns_for_validation for this check
            if col != '*': 
                invalid_columns.append(col)
    
    if invalid_columns:
        sorted_invalid_cols = sorted(list(set(invalid_columns)))
        # Only list tables that were actually found and used for validation context
        referenced_known_tables = [t for t in extracted_tables_lower if t in table_column_mapping]
        if not referenced_known_tables and extracted_tables_lower: 
            # This implies all tables were unknown, already handled by unknown_tables error.
            # This block might be redundant if unknown_tables is comprehensive.
            pass 
        
        table_context_msg = ""
        if referenced_known_tables:
            table_context_msg = f" for the referenced and defined tables [{', '.join(sorted(list(set(referenced_known_tables))))}]"
        elif extracted_tables_lower: # Some tables were mentioned but none were in DDL
             table_context_msg = f" (and referenced tables [{', '.join(sorted(extracted_tables_lower))}] were not defined in DDL)"
        else: # No tables mentioned in query FROM clause, but columns are being validated (should be caught earlier)
            table_context_msg = " (no tables referenced in FROM clause)"

        error_message = f"Columns [{', '.join(sorted_invalid_cols)}] are not defined{table_context_msg}"
        return False, [error_message]
    else:
        return True, []

# --- query_validator (Modified to handle potentially empty mapping better) ---
def query_validator(query, local_table_column_mapping):
    inspector = SQLQueryInspector(query)
    inspector_issues_msg = inspector.inspect_query() # Returns original query if no issues
    
    if inspector_issues_msg != query: # Issues were found by inspector
        return inspector_issues_msg
    
    # If local_table_column_mapping is empty (e.g., DDL parsing failed or was empty),
    # then any query with tables/columns will likely fail validation unless it's a "SELECT 1" type.
    if not local_table_column_mapping:
        # Check if it's a very simple query that doesn't need table/column mapping
        try:
            # A light parse attempt to see if it's table-less
            # We use a fresh parser instance as the main parser might fail on non-SELECT DML
            # that passed the inspector (e.g., if we allowed some "safe" DML for other purposes).
            # However, for column validation, we assume SELECT.
            temp_parser = Parser(query)
            if not temp_parser.tables and not temp_parser.columns_aliases and not temp_parser.columns:
                 # Truly simple, like "SELECT 1" or just comments
                 if re.match(r"\s*SELECT\s+[\d\.\s,'\"()*+-/]+(?:\s+AS\s+\w+)?\s*(?:;|$)", query, re.IGNORECASE | re.DOTALL):
                    return query # Simple select of constants/expressions
            
            # If there are tables/columns, but no mapping, it's an issue.
            # Except if it's a DML that passed inspector (e.g. "VACUUM;")
            # But column validation part will fail correctly.
            # Let's try to parse and if it refers to tables, it will fail validate_columns
        except Exception:
            pass # Ignore parsing error here, let the main logic proceed

    try:
        parser = Parser(query)
        
        # Handle cases where parser might not identify it as a SELECT-like query
        # for column/table extraction, especially if it's complex or dialect-specific.
        # `parser.query_type` can be 'SELECT', 'INSERT', etc.
        # We are mostly interested in validating columns for SELECT queries.
        if parser.query_type != "SELECT" and not parser.with_names: # Not a SELECT and not a CTE-based select
            # If the inspector passed it (e.g. no DML keywords like UPDATE, DELETE),
            # but it's not a SELECT, we assume it's valid from a column perspective
            # as column validation isn't applicable in the same way.
            # This could be a very simple command or expression.
            # Example: A query like "EXPLAIN SELECT 1" might pass inspector,
            # parser.query_type might be 'EXPLAIN'. We don't validate columns for it.
            if not parser.tables and not parser.columns: # If it also doesn't reference tables/columns
                return query 
            # If it's not SELECT but has tables/columns, it might be DDL/DML that inspector missed (unlikely for disallowed ones)
            # or something like "EXPLAIN ANALYZE SELECT..." - let it pass through for now if inspector was fine.
            # The goal is column validation for SELECTs.
            # For robust handling, one might want to explicitly whitelist query types.
            # For now, if not SELECT and passed inspector, consider it okay from column standpoint.
            # return query # Bypassing column validation for non-SELECTs that passed inspector.

        base_tables = [t.lower() for t in parser.tables if t] # Ensure t is not None or empty
        columns_raw = parser.columns if parser.columns else []
        ctes_present = bool(parser.with_names_normalized if hasattr(parser, 'with_names_normalized') else parser.with_names)
        
        # Handle simple SELECTs without FROM clause (e.g., SELECT 1, 'abc', NOW())
        if not base_tables and not ctes_present:
            is_simple_select = True
            if columns_raw:
                for c_token in columns_raw:
                    # A simple token can be a number, a string literal, a common function, or an aggregate.
                    # More robustly, check if it's an identifier that isn't a known function/keyword.
                    if not (c_token.isdigit() or
                            c_token == '*' or
                            agg_pattern.match(c_token) or
                            re.match(r'^\w+\(\s*(\w+\s*(,\s*\w+)*)?\s*\)$', c_token, re.IGNORECASE) or # my_func(), my_func(col)
                            re.match(r"'.*?'", c_token) or # 'string'
                            re.match(r'".*?"', c_token) or # "string"
                            re.match(r"^\d+(\.\d+)?$", c_token) or # 123 or 123.45
                            c_token.lower() in ['null', 'true', 'false', 'current_timestamp', 'current_date', 'current_time']
                           ):
                        is_simple_select = False
                        break
            if is_simple_select:
                return query # Valid simple select, no table/column schema validation needed.
            else:
                # Has columns but no tables, and not recognized as simple select constants/functions
                return "Validation Error: Columns specified without a valid table or CTE reference, and not a simple constant/function select."


        base_table_aliases = {alias.lower(): table.lower() for alias, table in parser.tables_aliases.items()}
        # Refine known_base_table_names to only those defined in DDL
        known_base_table_names_from_ddl = {t.lower() for t in base_tables if t.lower() in local_table_column_mapping}
        
        # Aliases that point to a table defined in DDL
        known_base_table_aliases_from_ddl = {
            alias for alias, table_name in base_table_aliases.items()
            if table_name in known_base_table_names_from_ddl
        }

        # If no tables from the query are actually defined in our DDL mapping,
        # but tables ARE mentioned in the query, this is an error.
        if base_tables and not known_base_table_names_from_ddl:
            # This check is now more robustly handled by validate_columns's "unknown_tables"
            pass

        columns_cleaned_for_validation = check_and_clean_columns(
            columns_raw, ctes_present, known_base_table_aliases_from_ddl, known_base_table_names_from_ddl
        )
        
        # Filter base_tables to only those that are known from DDL for validation context
        # This prevents trying to validate columns against tables that aren't defined.
        # The `validate_columns` function will report unknown tables separately.
        # For `validate_columns`, we pass all `base_tables` from query, it will sort out known/unknown.
        
        is_valid, validation_issues = validate_columns(
            base_tables, columns_cleaned_for_validation, local_table_column_mapping
        )
        
        if is_valid:
            return query
        else:
            return f"Validation Error: {', '.join(validation_issues)}"
            
    except Exception as e:
        # Log the full exception for debugging
        # logging.error(f"Error during query parsing or validation: {e}", exc_info=True)
        print(f"Error during query parsing or validation: {e}") # Keep print for CLI tests
        if "Unknown token" in str(e) or "Parse" in str(e) or "अनपेक्षित टोकन" in str(e): # Added Hindi for "Unexpected token"
            return f"Validation Error: Failed to parse the query structure. Check syntax. Error: ({e})"
        else:
            return f"Validation Error: An unexpected issue occurred during validation. Error: ({e})"


# --- Direct Test Execution Area ---
if __name__ == "__main__":
    print("--- Starting SQL Query Validation Tests ---")

    # Define DDL for our test schema
    test_ddl_scripts = """
    CREATE TABLE ecommerce_product_detail (
        product_id INT PRIMARY KEY,
        count INT,
        user_id VARCHAR(100),
        category VARCHAR(50),
        department VARCHAR(50),
        item_id STRING, -- Using STRING as a generic type
        assigned_to TEXT,
        tags ARRAY<STRING>, -- Example complex type, only name 'tags' matters
        priority VARCHAR(20),
        comments CLOB, -- Example large text type
        remediation_owner VARCHAR(100)
    );

    CREATE TABLE ecommerce_product_metadata (
        product_id INT,
        dataset_definition_id BIGINT,
        query_details TEXT,
        product_name VARCHAR(255),
        sql_filter VARCHAR(1000),
        product_metadata JSON,
        category VARCHAR(50),
        comments VARCHAR(500),
        priority VARCHAR(20),
        remediation_owner VARCHAR(100),
        FOREIGN KEY (product_id) REFERENCES ecommerce_product_detail(product_id)
    );

    CREATE TABLE order_view ( -- Naming it a view, but structure is like a table for parsing
        function_id INT,
        issue_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(100),
        function_info TEXT,
        issue_status VARCHAR(30),
        issue_description TEXT,
        issue_type VARCHAR(50),
        issue_level VARCHAR(20),
        occurrence_date DATE,
        report_date TIMESTAMP,
        importance INT
    );
    
    -- A table that won't be used in many successful queries to test "unknown table"
    CREATE TABLE IF NOT EXISTS unused_table (
        col_a INT,
        col_b TEXT
    );
    """

    print("\n--- Generating Table/Column Mapping from DDL ---")
    definitions_to_use = generate_table_column_mapping_from_ddl(test_ddl_scripts)
    if not definitions_to_use:
        print("CRITICAL: DDL parsing failed. Tests will not be meaningful.")
    else:
        print("Generated Mappings:")
        for table, cols in definitions_to_use.items():
            print(f"  {table}: {cols[:5]}..." if len(cols) > 5 else f"  {table}: {cols}")
    
    print("\n--- Running Test Cases ---")

    # Structure: [Query String, Expected Result ('Pass' or 'Fail'), Optional 'Reason Substring' for Fail]
    test_cases = [
        # === Valid Cases (based on DDL) ===
        ["SELECT product_id, product_name FROM ecommerce_product_metadata WHERE category = 'Electronics'", 'Pass'],
        ["SELECT * FROM order_view WHERE issue_level = 'High'", 'Pass'],
        ["SELECT ov.username, ov.issue_id FROM order_view ov WHERE ov.importance > 5", 'Pass'],
        ["SELECT product_id AS PID, product_name AS Name FROM ecommerce_product_metadata", 'Pass'],
        ["SELECT p.product_id, p.category FROM ecommerce_product_detail p WHERE p.department = 'Home'", 'Pass'],
        ["select product_id from ecommerce_product_metadata;", 'Pass'],
        ["SELECT 1", 'Pass'],
        ["SELECT 1, 'test', 3.14", 'Pass'],
        ["SELECT count(*) from ecommerce_product_detail", 'Pass'],
        ["SELECT COUNT(*) AS total_count FROM ecommerce_product_detail", 'Pass'],
        ["SELECT category, count(product_id) as NumberOfProducts FROM ecommerce_product_detail GROUP BY category", 'Pass'],
        ["SELECT current_timestamp", "Pass"], # Simple function/keyword like select

        # --- Valid JOIN Cases (based on DDL) ---
        ["SELECT o.username, p.product_name FROM order_view o JOIN ecommerce_product_metadata p ON o.username = p.remediation_owner", 'Pass'],
        ["SELECT epm.product_name, ov.issue_description FROM ecommerce_product_metadata epm LEFT JOIN order_view ov ON epm.product_id = ov.function_id WHERE epm.category = 'Tools'", 'Pass'],
        ["SELECT epd.item_id, epm.product_name, ov.issue_status FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm ON epd.product_id = epm.product_id JOIN order_view ov ON epm.remediation_owner = ov.username WHERE epd.department = 'Garden'", 'Pass'],
        ["SELECT p1.product_name, p2.product_name FROM ecommerce_product_metadata p1 JOIN ecommerce_product_metadata p2 ON p1.category = p2.category WHERE p1.product_id < p2.product_id", 'Pass'],
        ["SELECT epm.product_name, ov.username FROM ecommerce_product_metadata epm CROSS JOIN order_view ov WHERE epm.category = 'Books'", 'Pass'],
        ["SELECT epd.item_id, epm.product_name FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm USING (product_id)", 'Pass'],

        # --- Valid CTE Cases (columns from CTEs not validated against DDL, but base table columns are) ---
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail -- base table, cols 'category', 'product_id' are checked
             GROUP BY category
         )
         SELECT pc.category, pc.ProdCount -- pc.category, pc.ProdCount are from CTE
         FROM ProductCounts pc
         WHERE pc.ProdCount > 10
         """, 'Pass'],
        ["""
         WITH HighIssues AS (
             SELECT issue_id, username AS user_involved
             FROM order_view -- base table, cols 'issue_id', 'username' are checked
             WHERE issue_level = 'High'
         )
         SELECT epm.product_name, hi.user_involved
         FROM ecommerce_product_metadata epm -- base table, cols 'product_name', 'remediation_owner' are checked
         JOIN HighIssues hi ON epm.remediation_owner = hi.user_involved
         WHERE epm.priority = 'Critical'
         """, 'Pass'],


        # === Invalid Cases (Inspector Checks - DML/DDL/Injection) ===
        ["UPDATE ecommerce_product_metadata SET category = 'Outdoor' WHERE product_id = 1", 'Fail', "Potential disallowed operation detected: 'UPDATE'"],
        ["DELETE FROM order_view WHERE issue_id < 100", 'Fail', "Potential disallowed operation detected: 'DELETE'"],
        ["DROP TABLE ecommerce_product_detail", 'Fail', "Potential disallowed operation detected: 'DROP'"],
        ["CREATE INDEX idx_prod_id ON ecommerce_product_detail(product_id);", 'Fail', "Potential disallowed operation detected: 'CREATE'"],
        ["SELECT product_id FROM ecommerce_product_metadata; INSERT INTO order_view VALUES (1);", 'Fail', "Potential disallowed operation detected: 'INSERT'"],
        ["SELECT * FROM users WHERE username = 'admin' OR '1'='1'", 'Fail', "Potentially unsafe SQL pattern"], # Basic injection
        ["SELECT name FROM users WHERE id = '1' ; EXEC ('DROP TABLE users')", 'Fail', "Potentially unsafe SQL pattern 'EXEC ("], # EXEC
        ["SELECT xp_cmdshell('dir')", 'Fail', "Potentially unsafe SQL pattern 'xp_cmdshell'"],

        # === Cases that previously failed inspector but should now PASS inspector (and then pass/fail column validation) ===
        ["SELECT product_id FROM ecommerce_product_metadata LIMIT 5", 'Pass'], # LIMIT without ORDER BY is now OK for inspector
        ["SELECT product_id; SELECT username FROM order_view", 'Pass'], # Multiple statements separated by ; (parser will take first usually, inspector doesn't block based on this anymore)
                                                                         # For sql-metadata, this will likely parse as "SELECT product_id"
        ["SELECT p.product_name, u.username FROM ecommerce_product_metadata p JOIN order_view u", 'Pass'], # JOIN without ON/USING is now OK for inspector (sql-metadata usually treats as CROSS JOIN or dialect specific)
        ["SELECT product_id FROM ecommerce_product_detail UNION SELECT product_id FROM ecommerce_product_metadata", "Pass"], # UNION ok for inspector

        # === Invalid Cases (Column/Table/Parse Checks - based on DDL) ===
        ["SELECT product_id, non_existent_column FROM ecommerce_product_metadata", 'Fail', "Columns [non_existent_column] are not defined"],
        ["SELECT * FROM non_existent_table", 'Fail', "Query references undefined tables (not found in DDL): [non_existent_table]"],
        ["SELECT ov.username, ov.bad_column FROM order_view ov", 'Fail', "Columns [bad_column] are not defined"],
        ["SELECT user_id FROM user_data", 'Fail', "Query references undefined tables (not found in DDL): [user_data]"], # Assuming user_data not in DDL
        ["SELECT product_id FROM ecommerce_product_metadata m JOIN order_view o ON m.product_id = o.function_id WHERE o.invalid_col = 1", 'Fail', "Columns [invalid_col] are not defined"],
        ["SELECT COUNT(*) AS total_count, invalid_column FROM ecommerce_product_detail", 'Fail', "Columns [invalid_column] are not defined"],
        ["SELECT FROM ecommerce_product_detail", "Fail", "Failed to parse the query structure"], # Parse error
        ["SELECT col1 FROM ecommerce_product_detail JOIN non_existent_table nx ON nx.id = ecommerce_product_detail.product_id", "Fail", "Query references undefined tables (not found in DDL): [non_existent_table]"],

        # --- CTEs with invalid base column references ---
        ["""
         WITH ProductCounts AS (
             SELECT category_xyz, COUNT(product_id) as ProdCount -- category_xyz is invalid
             FROM ecommerce_product_detail
             GROUP BY category_xyz
         )
         SELECT pc.category_xyz, pc.ProdCount
         FROM ProductCounts pc
         """, 'Fail', "Columns [category_xyz] are not defined"],
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.ProdCount, epd.non_existent_base_column -- invalid column from base table
         FROM ProductCounts pc
         JOIN ecommerce_product_detail epd ON pc.category = epd.category
         """, 'Fail', "Columns [non_existent_base_column] are not defined"],
        # Limitation: Invalid column from CTE itself (not from base table) is not checked deeply against CTE definition by this validator
        ["""
         WITH ProductCounts AS (
             SELECT category, COUNT(product_id) as ProdCount
             FROM ecommerce_product_detail
             GROUP BY category
         )
         SELECT pc.category, pc.InvalidCTEName -- This column is from the CTE, not a base table
         FROM ProductCounts pc
         """, 'Pass'], # Current check_and_clean_columns doesn't validate CTE-generated columns against DDL

        # --- Invalid JOIN Cases (Column/Table Validation - based on DDL) ---
        ["SELECT epm.product_name FROM ecommerce_product_metadata epm JOIN order_view ov ON epm.product_id = ov.non_existent_key", 'Fail', "Columns [non_existent_key] are not defined"],
        ["SELECT epm.product_name, ov.invalid_column_from_join FROM ecommerce_product_metadata epm JOIN order_view ov ON epm.remediation_owner = ov.username", 'Fail', "Columns [invalid_column_from_join] are not defined"],
        ["SELECT p1.product_name, p2.invalid_self_col FROM ecommerce_product_metadata p1 JOIN ecommerce_product_metadata p2 ON p1.category = p2.category WHERE p1.product_id < p2.product_id", 'Fail', "Columns [invalid_self_col] are not defined"],
        ["SELECT epd.item_id, epm.product_name, ov.invalid_multi_join_col FROM ecommerce_product_detail epd JOIN ecommerce_product_metadata epm ON epd.product_id = epm.product_id JOIN order_view ov ON epm.remediation_owner = ov.username", 'Fail', "Columns [invalid_multi_join_col] are not defined"],

        # --- DDL Parsing Specific Tests (behavior of generate_table_column_mapping_from_ddl) ---
        # These are implicitly tested by above, but consider direct tests of generate_table_column_mapping_from_ddl if needed
        # For instance, an empty DDL should result in an empty mapping, causing most queries to fail table validation
        # ["SELECT * FROM some_table_not_in_empty_ddl", "Fail", "Query references undefined tables"] # if DDL was empty

        # --- Test case for query with only constants and no FROM clause (should pass) ---
        ["SELECT 'hello' as greeting, 123 as number, 45.67 as decimal_val", "Pass"],
        ["SELECT DATE('now')", "Pass"], # Function call without FROM
        ["SELECT * FROM ecommerce_product_detail WHERE product_id = 1; -- xp_cmdshell('dir')", 'Fail', "Potentially unsafe SQL pattern 'xp_cmdshell'"], # Injection check still active

    ]

    passed_count = 0
    failed_count = 0
    skipped_due_to_ddl_fail = 0

    if not definitions_to_use and any(tc[1] == 'Pass' or ("Columns [" in tc[2] if len(tc)>2 else False) for tc in test_cases) :
        print("\nWARNING: DDL mapping is empty. Column/Table validation tests will likely fail or be misleading.")
        # Optionally, one might choose to skip tests that rely heavily on DDL here.

    for i, test_case_data in enumerate(test_cases):
        query = test_case_data[0]
        expected_result = test_case_data[1]
        reason_substring = test_case_data[2] if len(test_case_data) > 2 else None

        print(f"\n--- Test Case {i+1} ---")
        print(f"Query  : {query.strip()}")
        print(f"Expect : {expected_result}" + (f" (Reason should contain: '{reason_substring}')" if reason_substring else ""))

        # Handle cases where DDL parsing might have failed completely
        if not definitions_to_use and expected_result == 'Pass' and not ("inspector" in query.lower() or "unsafe" in query.lower() or "disallowed" in query.lower()):
             # If DDL failed and test expects Pass based on schema, it's not a fair test of query_validator
             # This is a heuristic, better to ensure DDL parsing is robust.
             # For now, we'll run them, but they will likely fail if they need schema.
             pass


        actual_result_msg = query_validator(query, definitions_to_use)

        is_fail_result = "Detected issues" in actual_result_msg or "Validation Error" in actual_result_msg
        actual_outcome = 'Fail' if is_fail_result else 'Pass'
        
        print(f"Actual : {actual_outcome}")

        correct_outcome = (actual_outcome == expected_result)
        reason_check_passed = True

        if is_fail_result and expected_result == 'Fail' and reason_substring:
            if reason_substring.lower() not in actual_result_msg.lower():
                reason_check_passed = False
                print(f"Reason : MISMATCH! Expected reason substring '{reason_substring}' not found in actual error:")


        if correct_outcome and reason_check_passed:
            print("Result : CORRECT")
            passed_count += 1
        else:
            print(f"Result : INCORRECT ******")
            failed_count += 1
        
        if is_fail_result: # Always print reason if actual is Fail
            print(f"Reason : {actual_result_msg}")


    print("\n--- Validation Tests Summary ---")
    print(f"Total Cases: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    if skipped_due_to_ddl_fail > 0:
        print(f"Skipped due to DDL failure: {skipped_due_to_ddl_fail}")
    print("--- Testing Complete ---")
