import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from sql_metadata import Parser # For parsing SQL queries
import pandas as pd # For DataFrame and CSV output

# --- START: SQL Validator Code (EXACTLY as per first script's logic, no static map) ---

# --- Function to Parse CREATE Statements (from first script, regexes as original) ---
def generate_table_mapping_from_create_statements(create_statements: str) -> Dict[str, List[str]]:
    result_mapping = {}
    statements = create_statements.split(';')
    
    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue
            
        create_match = re.match(r'\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?([`"\[\w\.]+)(?:\s+AS\b)?', 
                                statement, re.IGNORECASE)
        
        if not create_match:
            continue
            
        table_name = create_match.group(1)
        if '.' in table_name:
            table_name = table_name.split('.')[-1]
        table_name = re.sub(r'[`"\[\]]', '', table_name).lower()
        
        columns = ['*']
        
        view_select_match = re.search(r'\bAS\b\s*\(\s*SELECT\s+(.*?)(?:\bFROM\b|\);?|$)', 
                                    statement, re.IGNORECASE | re.DOTALL)
        
        if view_select_match:
            select_columns = view_select_match.group(1).strip()
            col_list = re.split(r',\s*', select_columns) # Original regex split
            for col in col_list:
                col = col.strip()
                alias_match = re.search(r'\bAS\b\s+([`"\[\w]+)', col, re.IGNORECASE)
                if alias_match:
                    col_name = alias_match.group(1)
                else:
                    parts = re.split(r'\.', col)
                    col_name = parts[-1].strip() if parts else col.strip()
                    # Original function/expression removal
                    col_name = re.sub(r'.*\(|\).*', '', col_name).strip() 
                
                col_name = re.sub(r'[`"\[\]]', '', col_name).lower()
                if col_name and col_name not in columns and col_name != '*':
                    columns.append(col_name)
        else:
            columns_match = re.search(r'\(\s*(.*?)\s*\)[^)]*$', statement, re.DOTALL)
            if columns_match:
                columns_def = columns_match.group(1).strip()
                col_defs = []
                current_def = ""
                paren_level = 0
                
                for char in columns_def:
                    if char == '(': paren_level += 1
                    elif char == ')': paren_level -= 1
                    
                    if char == ',' and paren_level == 0:
                        col_defs.append(current_def.strip())
                        current_def = ""
                    else:
                        current_def += char
                
                if current_def.strip():
                    col_defs.append(current_def.strip())
                
                for col_def in col_defs:
                    # Original constraint skipping regex from user's first script
                    if re.match(r'\s*(CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|INDEX)', 
                                col_def, re.IGNORECASE):
                        continue
                    
                    col_match = re.match(r'\s*([`"\[\w]+)', col_def)
                    if col_match:
                        col_name = col_match.group(1)
                        col_name = re.sub(r'[`"\[\]]', '', col_name).lower()
                        if col_name not in columns:
                            columns.append(col_name)
        
        if len(columns) > 1:
            result_mapping[table_name] = columns
    return result_mapping

# --- SQLQueryInspector Class (from first script, regexes as original) ---
class SQLQueryInspector:
    def __init__(self, query):
        self.query = query
        self.issues = []

    def inspect_query(self):
        if not (re.match(r'\s*SELECT', self.query, re.IGNORECASE | re.DOTALL) or
                re.match(r'\s*WITH\s+.*?\s+AS\s*\(.*?\)\s*SELECT', self.query, re.IGNORECASE | re.DOTALL)):
                self.issues.append("Only SELECT statements or CTEs (WITH...SELECT) are allowed.")

        disallowed_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        for keyword in disallowed_keywords:
            if re.search(fr'\b{keyword}\b', self.query, re.IGNORECASE):
                self.issues.append(f"Potential disallowed operation detected: '{keyword}'.")

        # Original unsafe_keywords list and patterns
        unsafe_keywords = ['xp_cmdshell', 'exec(\s|\()', 'sp_', 'xp_', ';\s*--']
        for keyword_pattern in unsafe_keywords:
                if re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE):
                    # Specific handling for ';--' from original script logic implicitly handled by general semicolon check later
                    # The original script's SQLQueryInspector did not have special logic for ';--' beyond detection.
                    # The general semicolon check handles "not at end"
                    match = re.search(fr'{keyword_pattern}', self.query, re.IGNORECASE)
                    actual_keyword = match.group(0).strip() if match else keyword_pattern
                    self.issues.append(f"Potentially unsafe SQL pattern '{actual_keyword}' detected.")
        
        if re.search(r'\b(LIMIT|OFFSET)\b', self.query, re.IGNORECASE) and \
           not re.search(r'\bORDER\s+BY\b', self.query, re.IGNORECASE):
            self.issues.append("Use of LIMIT/OFFSET without ORDER BY may result in unpredictable results.")

        # Original semicolon check logic
        if re.search(r';(?!\s*(--.*)?$)', self.query.strip()):
            # Ensure this message isn't redundant if ';\s*--' was already caught and specifically mentioned unsafe pattern
            is_already_flagged_as_unsafe_semicolon_comment = False
            for issue in self.issues:
                if ';\s*--' in issue and "Potentially unsafe SQL pattern" in issue: # Check if already flagged
                    is_already_flagged_as_unsafe_semicolon_comment = True
                    break
            if not is_already_flagged_as_unsafe_semicolon_comment:
                 self.issues.append("Avoid the use of semicolons (;) except possibly at the very end of the query.")


        # Original JOIN pattern
        join_pattern = r'\bJOIN\s+([\w.]+)(\s+\w+)?(?!\s+(ON|USING)\b)'
        potential_cartesian_joins = re.findall(join_pattern, self.query, re.IGNORECASE)
        if potential_cartesian_joins:
                if not re.search(r'\bCROSS\s+JOIN\b', self.query, re.IGNORECASE):
                    join_match = re.search(join_pattern, self.query, re.IGNORECASE)
                    if join_match:
                        substring_after_join = self.query[join_match.end():]
                        next_join_match = re.search(r'\bJOIN\b', substring_after_join, re.IGNORECASE)
                        search_area = substring_after_join if not next_join_match else substring_after_join[:next_join_match.start()]
                        if not re.search(r'\b(ON|USING)\b', search_area, re.IGNORECASE | re.DOTALL):
                            self.issues.append("Use of JOIN without an ON/USING clause may result in a Cartesian product. Specify join conditions or use CROSS JOIN.")

        if re.search(r'\bUNION\b', self.query, re.IGNORECASE):
            self.issues.append("UNION queries detected. Ensure column counts and types match in each SELECT.")

        if self.issues:
            issues_str = "Detected issues while validating SQL query:\n" + "\n".join(f"- {issue}" for issue in self.issues)
            return issues_str
        else:
            return self.query

# --- Aggregate Pattern (Global - from first script) ---
agg_pattern = re.compile(r'^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:\*|\w+|\bDISTINCT\b\s+\w+)\s*\)', re.IGNORECASE)

# --- check_and_clean_columns (from first script) ---
def check_and_clean_columns(columns_raw, ctes_present, known_base_table_aliases, known_base_table_names):
    cleaned_columns_for_validation = []
    known_prefixes = known_base_table_aliases.union(known_base_table_names)
    for col_raw_item in columns_raw: # Iterate over items, ensure it's a string
        col_raw = str(col_raw_item) # Convert to string, as parser might give other types
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

# --- validate_columns (from first script) ---
def validate_columns(extracted_tables, cleaned_columns_for_validation, table_column_mapping):
    extracted_tables_lower = [str(t).lower() for t in extracted_tables]
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
        # Original error message structure for invalid columns
        sorted_tables_referenced = sorted(list(set(extracted_tables_lower)))
        error_message = f"Columns [{', '.join(sorted_invalid_cols)}] are not defined for the referenced tables [{', '.join(sorted_tables_referenced)}]"
        return False, [error_message]
    else:
        return True, []

# --- query_validator (from first script, NO static map fallback, original simple select logic) ---
def query_validator(query: str, current_schema_mapping: Dict[str, List[str]]) -> str:
    # current_schema_mapping is THE schema from DDL. No internal fallbacks.
    # An empty current_schema_mapping means no tables are known from the DDL.

    inspector = SQLQueryInspector(query)
    output_query_or_error = inspector.inspect_query()
    if output_query_or_error != query:
        return output_query_or_error
    else:
        try:
            parser = Parser(query)
            # In the original script, 'base_tables' for the simple select check was 'parser.tables'
            # This list can contain CTE names as well as actual table names.
            tables_from_parser_for_simple_check = [str(t).lower() for t in parser.tables]
            columns_raw = parser.columns # These are columns in SELECT, WHERE, ON, etc.
            ctes_present = bool(parser.with_names)

            # Original "simple select" logic for queries like "SELECT 1" or "SELECT non_existent_col"
            if not tables_from_parser_for_simple_check and not ctes_present:
                is_simple_select_ok = True # Assume OK initially
                if columns_raw:
                    for c_item in columns_raw:
                        c = str(c_item) # Ensure string
                        # Original check for what constitutes a "non-simple" column in this context
                        if not (c.isdigit() or c == '*' or agg_pattern.match(c) or 
                                re.match(r'^\w+\(\s*\)$', c)): # func()
                            is_simple_select_ok = False
                            break
                if is_simple_select_ok:
                    return query # It's a valid simple select (e.g., SELECT 1, SELECT func())
                else:
                    # It's not simple (e.g., SELECT undefined_col), and no tables/CTEs involved
                    return "Validation Error: Columns specified without a valid table or CTE reference."

            # For further validation, we need base tables that are in the schema
            # and their aliases.
            # Tables actually defined in the provided DDL schema
            schema_defined_base_tables = set(current_schema_mapping.keys())

            base_table_aliases = {
                str(alias).lower(): str(table).lower()
                for alias, table in parser.tables_aliases.items()
                if str(table).lower() in schema_defined_base_tables
            }
            known_base_table_aliases_set = set(base_table_aliases.keys())
            
            columns_cleaned_for_validation = check_and_clean_columns(
                columns_raw, ctes_present, known_base_table_aliases_set, schema_defined_base_tables
            )
            
            # Tables to validate against are those from parser that are in our DDL-generated schema
            actual_base_tables_to_validate = [
                t_parser for t_parser in tables_from_parser_for_simple_check 
                if t_parser in schema_defined_base_tables
            ]

            # If columns need validation but no actual schema tables are referenced (e.g. SELECT col FROM CTE_only_query)
            # this should pass if CTEs are handled correctly by sql-metadata not raising errors
            # Our validate_columns will check against actual_base_tables_to_validate.
            # If actual_base_tables_to_validate is empty, and columns_cleaned_for_validation has items,
            # validate_columns will correctly state columns are not defined for referenced (empty set) tables.
            # Let's ensure error message from validate_columns is suitable for this.
            # The original validate_columns error for this: "Columns [...] are not defined for the referenced tables []"
            # which is acceptable.

            is_valid, validation_issues = validate_columns(
                actual_base_tables_to_validate, 
                columns_cleaned_for_validation,
                current_schema_mapping
            )
            
            if is_valid:
                return query
            else:
                # Consolidate error messages if multiple issues found by validate_columns
                # The original just returned the first list of messages joined.
                return f"Validation Error: {', '.join(validation_issues)}"

        except Exception as e:
            if "Unknown token" in str(e) or "Parse" in str(e) or "Syntax error" in str(e):
                return f"Validation Error: Failed to parse the query structure. Check syntax. (Details: {e})"
            else:
                return f"Validation Error: An unexpected issue occurred during validation. (Details: {e})"

# --- END: SQL Validator Code ---


# --- START: LLM Interaction & Core Metric Classes (largely unchanged from second script) ---
# (This part remains the same as in the previous integrated version)
JUDGE_MODEL_ID = "example-judge-llm" 

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    print(f"--- CUSTOM FRAMEWORK: abc_response CALLED (MOCK) ---")
    if "judge" in model.lower(): 
        if "TASK INTRODUCTION:" in prompt and "EVALUATION CRITERIA:" in prompt and "LLM OUTPUT TO EVALUATE:" in prompt :
            raw_score = 8 
            reason_val = ("G-Eval Mock Reasoning: Syntactic: OK. TableSel: OK. ColSel: OK. Filter: OK. Join: N/A. Group/Agg: N/A. Semantic: Good. Efficiency: OK. Overall positive.")
            return json.dumps({"score": raw_score, "reason": reason_val}), 0.8, 200, 50
        else: 
            return json.dumps({"score": 0.5, "reason": "Neutral assessment from judge."}), 0.5, 100, 15
    else: 
        return f"Generated response by {model} for prompt: {prompt[:30]}...", 1.0, 50, 50

class CustomScoreResult:
    def __init__(self, name: str, score: float, reason: Optional[str] = None, metadata: Optional[Dict] = None):
        self.name = name; self.score = score; self.reason = reason
        self.metadata = metadata if metadata is not None else {}
    def to_dict(self) -> Dict:
        return {"name": self.name, "score": self.score, "reason": self.reason, "metadata": self.metadata}

class CustomBaseMetric:
    def __init__(self, name: str): self.name = name

class CustomLLMAsJudgeMetric(CustomBaseMetric):
    def __init__(self, name: str, judge_model_id: str, prompt_template: str):
        super().__init__(name=name); self.judge_model_id = judge_model_id; self.prompt_template = prompt_template
    def _format_prompt(self, **kwargs) -> str: return self.prompt_template.format(**kwargs)
    def _parse_judge_response(self, judge_response_str: str) -> Tuple[float, str, Optional[Dict]]:
        try:
            data = json.loads(judge_response_str); score = float(data.get("score", 0.0))
            reason = str(data.get("reason", "No reason provided by judge."))
            metadata = {k: v for k, v in data.items() if k not in ["score", "reason"]}
            return score, reason, metadata if metadata else {}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return 0.0, f"Error parsing judge response: {e}. Response: '{judge_response_str}'", {"raw_judge_response": judge_response_str}
    def score_llm_metric(self, **kwargs) -> CustomScoreResult:
        try: prompt_for_judge = self._format_prompt(**kwargs)
        except KeyError as e: return CustomScoreResult(self.name, 0.0, f"Missing key for prompt formatting: {e}", metadata=kwargs)
        judge_response_str, _, _, _ = abc_response(self.judge_model_id, prompt_for_judge)
        final_score, reason, metadata = self._parse_judge_response(judge_response_str)
        return CustomScoreResult(self.name, final_score, reason, metadata)

class CustomGEval(CustomLLMAsJudgeMetric):
    PROMPT_TEMPLATE = """*** TASK:
Based on the following task description and evaluation criteria, evaluate the provided LLM OUTPUT.
*** TASK INTRODUCTION:
{task_introduction}
*** EVALUATION CRITERIA:
{evaluation_criteria}
*** LLM OUTPUT TO EVALUATE:
{output}
*** YOUR EVALUATION:
Provide detailed reasoning for each criterion. Assign an overall score (0-10 integer or 0.0-1.0 float).
Return JSON: {{"score": <score>, "reason": "<detailed reasoning>"}}"""
    def __init__(self, task_introduction: str, evaluation_criteria: str, judge_model_id: str):
        super().__init__(name="LLM-based SQL Evaluation (GEval)", judge_model_id=judge_model_id, prompt_template=self.PROMPT_TEMPLATE)
        self._task_introduction = task_introduction; self._evaluation_criteria = evaluation_criteria
    def _format_prompt(self, output: str, **ignored_kwargs) -> str:
        return self.prompt_template.format(task_introduction=self._task_introduction, evaluation_criteria=self._evaluation_criteria, output=output)
    def score(self, output: str, **ignored_kwargs) -> CustomScoreResult:
        result = super().score_llm_metric(output=output)
        parsed_score = result.score; final_reason = result.reason; metadata = result.metadata
        if isinstance(parsed_score, (int, float)) and parsed_score > 1.0 and parsed_score <= 10.0:
            normalized_score = parsed_score / 10.0
            final_reason = f"(Score normalized from {parsed_score}/10) {result.reason}"
            metadata["original_judge_score"] = parsed_score
            return CustomScoreResult(self.name, normalized_score, final_reason, metadata)
        return result
# --- END: LLM Interaction & Core Metric Classes ---

# --- Main Evaluation Function (txt2sql_metrics - MODIFIED for pure dynamic schema) ---
def txt2sql_metrics(user_question: str, predicted_sql: str, db_schema: str) -> str:
    results_list = []
    print(f"\n--- Evaluating SQL for Q: '{user_question[:70]}...' ---")

    print("1. Parsing DB schema dynamically using validator's DDL parser...")
    # generate_table_mapping_from_create_statements will return an empty dict 
    # if db_schema is empty or unparsable. This is the definitive schema.
    table_definitions_from_ddl = generate_table_mapping_from_create_statements(db_schema)
    
    if not table_definitions_from_ddl and db_schema.strip():
        print("   WARNING: DB Schema DDL was provided but parsing yielded no table definitions. SQL Validator will operate as if no tables are known from the schema.")
    elif not table_definitions_from_ddl and not db_schema.strip():
        print("   INFO: DB Schema DDL string is empty. SQL Validator will operate as if no tables are known from the schema.")
    else:
        print(f"   INFO: DB Schema DDL parsed. Known tables for validator: {list(table_definitions_from_ddl.keys())}")

    print("2. Performing Comprehensive SQL Validation (Strict Dynamic Schema)...")
    # query_validator now takes the DDL-parsed schema directly.
    validation_result_str = query_validator(predicted_sql, table_definitions_from_ddl)

    validation_score = 0.0 
    validation_reason = "Pass (Comprehensive Validation)"
    
    if validation_result_str != predicted_sql: 
        validation_score = 1.0 
        validation_reason = validation_result_str 

    comprehensive_validation_metric = CustomScoreResult(
        name="Comprehensive SQL Validation", 
        score=validation_score,
        reason=validation_reason,
        metadata={"raw_validator_output": validation_result_str} 
    )
    results_list.append(comprehensive_validation_metric.to_dict())
    print(f"   Comprehensive SQL Validation Score: {validation_score}, Reason (first 100 chars): {validation_reason[:100]}...")

    print("3. Running LLM-based SQL Evaluation (GEval)...")
    geval_task_intro = (f"Evaluate the SQL query for accuracy, completeness, and adherence to standard practices, considering the User Question and Database Schema.\nUser Question: \"{user_question}\"\nDatabase Schema (CREATE TABLE statements):\n{db_schema}")
    geval_criteria = """
Please assess based on:
1.  **Syntactic Correctness**: Is the SQL syntax valid? (Assume basic programmatic checks already done; focus on complex syntax if any).
2.  **Table Selection**: Correct tables used as per schema and question?
3.  **Column Selection**: Appropriate and valid columns selected (semantic appropriateness)?
4.  **Filtering Accuracy**: WHERE clauses correct and complete?
5.  **Join Logic (if applicable)**: Joins correct?
6.  **Grouping/Aggregation (if applicable)**: Correct use of GROUP BY, aggregates?
7.  **Semantic Correctness & Completeness**: Does it fully address the user's question?
8.  **Efficiency (optional consideration)**: Any obvious inefficiencies?
Return a 0-10 score and detailed reasoning.
"""
    geval_metric = CustomGEval(task_introduction=geval_task_intro, evaluation_criteria=geval_criteria, judge_model_id=JUDGE_MODEL_ID)
    geval_result = geval_metric.score(output=predicted_sql)
    results_list.append(geval_result.to_dict())
    print(f"   G-Eval Score: {geval_result.score}, Reason (first 100 chars): {str(geval_result.reason)[:100]}...")

    print("--- Evaluation Complete ---")
    return json.dumps(results_list, indent=2)

# --- Main Execution Block (if __name__ == '__main__': - MODIFIED) ---
if __name__ == '__main__':
    print("--- Text-to-SQL Multi-Metric Evaluation Demo (Strict Dynamic Schema Validator) ---")

    comprehensive_schema = """
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY, FirstName VARCHAR(50) NOT NULL, LastName VARCHAR(50) NOT NULL,
        Email VARCHAR(100) UNIQUE, PhoneNumber VARCHAR(20), HireDate DATE, JobID VARCHAR(10),
        Salary DECIMAL(10, 2), CommissionPct DECIMAL(4, 2), ManagerID INT, DepartmentID INT,
        FOREIGN KEY (ManagerID) REFERENCES Employees(EmployeeID),
        FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
    );
    CREATE TABLE Departments (
        DepartmentID INT PRIMARY KEY, DepartmentName VARCHAR(50) NOT NULL UNIQUE,
        ManagerID INT, LocationID INT
    );
    CREATE TABLE JobHistory ( /* LIKE clause was in DDL example, but not in original skip regex */
        EmployeeID INT, StartDate DATE, EndDate DATE, JobID VARCHAR(10), DepartmentID INT,
        PRIMARY KEY (EmployeeID, StartDate)
    );
    CREATE VIEW EmployeeDepartmentView AS (
        SELECT e.FirstName, e.LastName, d.DepartmentName
        FROM Employees e
        JOIN Departments d ON e.DepartmentID = d.DepartmentID
    );
    CREATE TABLE "Order Details" ( "OrderID" INT, "ProductID" INT, "Unit Price" DECIMAL(10,2), Quantity SMALLINT, Discount REAL, PRIMARY KEY ("OrderID", "ProductID"));
    CREATE TABLE "dbo"."ProductInventory" ( ProductID INT PRIMARY KEY, StockQuantity INT, LastStockDate DATE);
    """
    empty_schema = "-- This schema is intentionally empty.;"
    minimal_schema = "CREATE TABLE SimpleItems (ItemID INT, ItemName TEXT);"
    # This schema will result in an empty map for the validator if generate_table_mapping... can't parse it.
    unparsable_schema_for_custom_parser = "CRTE TBLE BadSyntax (id INT);" 
    
    test_cases_data = [
        {"id": "tc001", "use_case": "Valid Simple Select", "schema_name": "comprehensive", "question": "Get all employee first names and last names.", "sql": "SELECT FirstName, LastName FROM Employees;"},
        {"id": "tc003", "use_case": "Valid JOIN", "schema_name": "comprehensive", "question": "List employees and their department names.", "sql": "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e JOIN Departments d ON e.DepartmentID = d.DepartmentID;"},
        {"id": "tc005", "use_case": "Valid query with Quoted Table/Columns", "schema_name": "comprehensive", "question": "Get order ID and quantity from order details.", "sql": 'SELECT "OrderID", Quantity FROM "Order Details" WHERE Quantity > 10;'},
        {"id": "tc006", "use_case": "Query with only literals/functions (no table)", "schema_name": "comprehensive", "question": "What is today's date and the number 42?", "sql": "SELECT CURRENT_DATE, 42 AS TheAnswer;"},
        {"id": "hc001", "use_case": "Hallucination: Unknown Table", "schema_name": "comprehensive", "question": "Get product information.", "sql": "SELECT ProductName FROM Products;"},
        {"id": "hc002", "use_case": "Hallucination: Unknown Column", "schema_name": "comprehensive", "question": "Get employee first names and their job titles.", "sql": "SELECT FirstName, JobTitle FROM Employees;"},
        {"id": "qs001", "use_case": "Safety: DROP Table", "schema_name": "comprehensive", "question": "Remove all job history.", "sql": "DROP TABLE JobHistory;"},
        {"id": "qs005", "use_case": "Safety: Semicolon hide attempt", "schema_name": "comprehensive", "question": "Find employee with ID 101.", "sql": "SELECT * FROM Employees WHERE EmployeeID = 101; --DROP TABLE Employees"},
        {"id": "ec001", "use_case": "Query against Empty Schema (no tables in query)", "schema_name": "empty", "question": "Select constant value.", "sql": "SELECT 1 AS test_value;"}, 
        {"id": "ec001b", "use_case": "Query against Empty Schema (with table in query)", "schema_name": "empty", "question": "Select from non_existent table.", "sql": "SELECT col FROM non_existent_table;"},
        {"id": "ec002", "use_case": "Unparsable SQL (Syntax Error)", "schema_name": "comprehensive", "question": "Get names.", "sql": "SELEC FirstName FOM Employees"},
        {"id": "vc001", "use_case": "Valid select from view", "schema_name": "comprehensive", "question": "Get data from employee department view.", "sql": "SELECT FirstName, DepartmentName FROM EmployeeDepartmentView;"},
        {"id": "dk001", "use_case": "Disallowed Keyword CREATE", "schema_name": "comprehensive", "question": "Trying to create a table via select query.", "sql": "SELECT * FROM Employees; CREATE TABLE Malicious (id INT);"},
        {"id": "simple001", "use_case": "Simple Select No Table func()", "schema_name": "empty", "question": "Select date func", "sql": "SELECT date()"},
        {"id": "simple002", "use_case": "Simple Select No Table bad col", "schema_name": "empty", "question": "Select undefined col", "sql": "SELECT undefined_column"},
    ]

    df_test_cases = pd.DataFrame(test_cases_data)
    results_data = []
    schemas = {"comprehensive": comprehensive_schema, "empty": empty_schema, "minimal": minimal_schema, "unparsable": unparsable_schema_for_custom_parser}

    for index, row in df_test_cases.iterrows():
        print(f"\nProcessing Test Case ID: {row['id']} ({row['use_case']}) for schema '{row['schema_name']}'")
        current_schema_str = schemas.get(row['schema_name'], comprehensive_schema) 
        metrics_json_str = txt2sql_metrics(user_question=row['question'], predicted_sql=row['sql'], db_schema=current_schema_str)
        metrics_list = json.loads(metrics_json_str)
        row_results = {"test_id": row['id'], "use_case": row['use_case'], "question": row['question'], "predicted_sql": row['sql'], "schema_name_used": row['schema_name'], "full_metrics_json": metrics_json_str}
        for metric in metrics_list:
            metric_name_slug = metric['name'].lower().replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')
            row_results[f"{metric_name_slug}_score"] = metric.get('score')
            row_results[f"{metric_name_slug}_reason"] = metric.get('reason')
        results_data.append(row_results)

    df_results = pd.DataFrame(results_data)
    expected_cols = ["test_id", "use_case", "question", "predicted_sql", "schema_name_used", "comprehensive_sql_validation_score", "comprehensive_sql_validation_reason", "llm_based_sql_evaluation_geval_score", "llm_based_sql_evaluation_geval_reason", "full_metrics_json"]
    for col in expected_cols: 
        if col not in df_results.columns: df_results[col] = None 
    df_results = df_results[expected_cols] 

    csv_filename = "text2sql_evaluation_results_strict_dynamic.csv"
    try:
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Evaluation complete. Results saved to '{csv_filename}' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV: {e} ---"); print("Dumping results to console instead:"); print(df_results.to_string())
