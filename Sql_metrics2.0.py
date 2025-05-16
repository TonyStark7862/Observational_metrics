import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from sql_metadata import Parser # For parsing SQL queries
import pandas as pd # For DataFrame and CSV output

# --- START: LLM Interaction & Core Metric Classes (largely unchanged) ---

JUDGE_MODEL_ID = "example-judge-llm"

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    print(f"--- CUSTOM FRAMEWORK: abc_response CALLED ---")
    print(f"Model: {model}")
    if len(prompt) > 300: # Truncate long prompts for cleaner logs
        print(f"Prompt (truncated):\n{prompt[:300]}...\n--------------------------------------")
    else:
        print(f"Prompt:\n{prompt}\n--------------------------------------")

    if "judge" in model.lower():
        if "TASK INTRODUCTION:" in prompt and "EVALUATION CRITERIA:" in prompt: # G-Eval
            raw_score = 8 # Mock score for G-Eval
            reason_val = (
                "G-Eval Mock Reasoning: Syntactic: OK. TableSel: OK. ColSel: OK. Filter: OK. Join: N/A. Group/Agg: N/A. Semantic: Good. Efficiency: OK. Overall positive."
            )
            return json.dumps({"score": raw_score, "reason": reason_val}), 0.8, 200, 50
        else: # Default judge response for other metric types (not used in this version)
            return json.dumps({"score": 0.5, "reason": "Neutral assessment from judge."}), 0.5, 100, 15
    else: # Primary LLM response (not used by judge metrics)
        return f"Generated response by {model} for prompt: {prompt[:30]}...", 1.0, 50, 50

class CustomScoreResult:
    def __init__(self, name: str, score: float, reason: Optional[str] = None, metadata: Optional[Dict] = None):
        self.name = name
        self.score = score # 0.0 generally means pass/good, 1.0 means fail/bad for heuristic checks
        self.reason = reason
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata
        }

class CustomBaseMetric: # Simplified, as only CustomGEval uses full features here
    def __init__(self, name: str):
        self.name = name

class CustomLLMAsJudgeMetric(CustomBaseMetric):
    def __init__(self, name: str, judge_model_id: str, prompt_template: str):
        super().__init__(name=name)
        self.judge_model_id = judge_model_id
        self.prompt_template = prompt_template

    def _format_prompt(self, **kwargs) -> str:
        return self.prompt_template.format(**kwargs)

    def _parse_judge_response(self, judge_response_str: str) -> Tuple[float, str, Optional[Dict]]:
        try:
            data = json.loads(judge_response_str)
            score = float(data.get("score", 0.0)) # Default to 0.0 if score is missing
            reason = str(data.get("reason", "No reason provided by judge."))
            metadata = {k: v for k, v in data.items() if k not in ["score", "reason"]}
            return score, reason, metadata if metadata else {}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return 0.0, f"Error parsing judge response: {e}. Response: '{judge_response_str}'", {"raw_judge_response": judge_response_str}

    def score_llm_metric(self, **kwargs) -> CustomScoreResult: # Renamed to avoid conflict
        try:
            prompt_for_judge = self._format_prompt(**kwargs)
        except KeyError as e: # Should not happen if all inputs to format are provided
            return CustomScoreResult(self.name, 0.0, f"Missing key for prompt formatting: {e}", metadata=kwargs)

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
        self._task_introduction = task_introduction
        self._evaluation_criteria = evaluation_criteria

    def _format_prompt(self, output: str, **ignored_kwargs) -> str:
        return self.prompt_template.format(
            task_introduction=self._task_introduction,
            evaluation_criteria=self._evaluation_criteria,
            output=output
        )

    def score(self, output: str, **ignored_kwargs) -> CustomScoreResult: # score method for CustomGEval
        result = super().score_llm_metric(output=output) # Calls the renamed parent method
        parsed_score = result.score
        final_reason = result.reason
        metadata = result.metadata

        if isinstance(parsed_score, (int, float)) and parsed_score > 1.0 and parsed_score <= 10.0: 
            normalized_score = parsed_score / 10.0
            final_reason = f"(Score normalized from {parsed_score}/10) {result.reason}"
            metadata["original_judge_score"] = parsed_score
            return CustomScoreResult(self.name, normalized_score, final_reason, metadata)
        return result
# --- END: LLM Interaction & Core Metric Classes ---

# --- START: Heuristic SQL Checks (Schema Adherence, Safety) ---

DDL_STATEMENT_REGEX = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([a-zA-Z0-9_.\"\[\]]+)\s*\((.*?)\)\s*;?",
    re.IGNORECASE | re.DOTALL
)
COLUMN_NAME_CAPTURE_REGEX = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])')

def parse_db_schema(db_schema_str: str) -> Dict[str, List[str]]:
    print("--- Starting DB Schema Parsing ---")
    table_column_map = {}
    # Remove block comments first to simplify line-by-line parsing later
    db_schema_str_no_blocks = re.sub(r'/\*.*?\*/', '', db_schema_str, flags=re.DOTALL)

    for ddl_match in DDL_STATEMENT_REGEX.finditer(db_schema_str_no_blocks):
        table_name_raw = ddl_match.group(1).strip()
        columns_text_raw = ddl_match.group(2)
        
        print(f"\n  Processing DDL for raw table string: '{table_name_raw}'")
        print(f"  Raw columns block:\n'''\n{columns_text_raw[:200]}...\n'''") # Print start of columns block

        # Normalize table name: lowercase, remove quotes/brackets, take last part if schema.table
        table_name_parts = table_name_raw.split('.')
        table_name_normalized = table_name_parts[-1].lower().strip('"[]`')
        
        if not table_name_normalized: # Should not happen if regex matches well
            print(f"    WARNING: Could not derive normalized table name from '{table_name_raw}'")
            continue
            
        print(f"    Normalized Table Name: '{table_name_normalized}'")

        current_columns = ['*'] # Always allow '*' wildcard
        # Remove line comments from columns_text
        lines = []
        for line in columns_text_raw.split('\n'):
            lines.append(re.sub(r'--.*', '', line).strip())
        columns_text_cleaned = "\n".join(lines)

        # Split column definitions by comma, carefully handling parentheses
        balance = 0
        current_segment = ""
        column_segments = []
        for char in columns_text_cleaned:
            if char == '(': balance += 1
            elif char == ')': balance -=1
            
            if char == ',' and balance == 0: # Split by comma only if not inside parentheses
                column_segments.append(current_segment.strip())
                current_segment = ""
            else:
                current_segment += char
        column_segments.append(current_segment.strip()) # Add the last segment

        extracted_column_names = []
        for segment in column_segments:
            segment = segment.strip()
            if not segment or segment.upper().startswith((
                'CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 
                'UNIQUE', 'CHECK', 'INDEX', 'LIKE' # LIKE for table inheritance/templating
            )):
                continue # Skip constraint definitions or other non-column lines

            col_match = COLUMN_NAME_CAPTURE_REGEX.match(segment)
            if col_match:
                col_name_raw = col_match.group(1)
                col_name_normalized = col_name_raw.lower().strip('"[]`') # Normalize column name
                if col_name_normalized: # Ensure not empty after stripping
                    extracted_column_names.append(col_name_normalized)
        
        current_columns.extend(list(set(extracted_column_names))) # Add unique extracted names
        table_column_map[table_name_normalized] = list(set(current_columns)) # Ensure unique columns in final list
        print(f"    Extracted Normalized Columns for '{table_name_normalized}': {table_column_map[table_name_normalized]}")
    
    if not table_column_map:
        print("  WARNING: Schema parsing resulted in an empty map (no tables/columns found).")
    print("--- DB Schema Parsing Complete ---")
    return table_column_map


def check_sql_column_hallucination(predicted_sql: str, schema_map: Dict[str, List[str]]) -> CustomScoreResult:
    name = "SQL Column Hallucination" # Name as per user request
    metadata = {"details": []}

    # Handle case where schema_map is empty (e.g., unparsable schema or literally empty DDL)
    if not schema_map:
        try:
            parser_for_empty_schema_check = Parser(predicted_sql)
            # If query tries to access any tables, it's a hallucination against an empty/unparsed schema
            if parser_for_empty_schema_check.tables:
                reason = "Query targets tables, but the provided DDL schema is effectively empty or could not be parsed."
                metadata["details"].append(reason)
                return CustomScoreResult(name=name, score=1.0, reason=reason, metadata=metadata)
            else: # Query is simple (e.g. SELECT 1) and schema is empty, so no hallucination of tables/columns
                return CustomScoreResult(name=name, score=0.0, reason="Pass", metadata=metadata)
        except Exception as e_parse_sql: # Query itself is unparsable
            reason = f"Pass (Query unparsable: {str(e_parse_sql)}, cannot determine hallucination against empty/unparsed schema)."
            metadata["details"].append(f"Query parsing error for hallucination check: {str(e_parse_sql)}")
            return CustomScoreResult(name=name, score=0.0, reason=reason, metadata=metadata)

    try:
        parser = Parser(predicted_sql)
        queried_tables_raw = parser.tables
        # queried_columns_raw = parser.columns # Using columns_dict is generally better
    except Exception as e:
        metadata["details"].append(f"SQL parsing error: {str(e)}")
        # If query is unparsable, it cannot adhere to schema definitions.
        return CustomScoreResult(name=name, score=1.0, reason=f"Query unparsable, cannot check for column hallucination: {str(e)}", metadata=metadata)

    # Normalize queried table names (lowercase, strip quotes, take last part of schema.table)
    queried_tables_normalized = list(set([t.split('.')[-1].lower().strip('"[]`') for t in queried_tables_raw if t]))
    
    unknown_tables_found = []
    for qt_normalized in queried_tables_normalized:
        if qt_normalized not in schema_map:
            unknown_tables_found.append(qt_normalized)
    
    if unknown_tables_found:
        reason = f"Query references undefined tables: {', '.join(list(set(unknown_tables_found)))}."
        metadata["details"].append(reason)
        return CustomScoreResult(name=name, score=1.0, reason=reason, metadata=metadata)

    # If no tables are queried (e.g. SELECT 1+1), then no column hallucination relative to tables.
    if not queried_tables_normalized and parser.columns_dict and not any(parser.columns_dict.values()): # e.g. SELECT 1
         return CustomScoreResult(name=name, score=0.0, reason="Pass (No tables queried, simple select)", metadata=metadata)
    if not queried_tables_normalized and not parser.tables: # Truly no tables and parser.columns_dict might be empty for SELECT 1
        is_simple_select = True
        if parser.columns:
             for col_item in parser.columns:
                 if isinstance(col_item, str) and not (col_item.isnumeric() or '(' in col_item or col_item.upper() in ['CURRENT_DATE', 'NOW()']): # very basic check
                     is_simple_select = False; break
        if is_simple_select:
            return CustomScoreResult(name=name, score=0.0, reason="Pass (No tables queried, simple select)", metadata=metadata)


    # Aggregate all valid column names from the tables actually queried and present in the schema_map
    valid_columns_for_queried_schema = set(['*']) # '*' is always permitted
    for qt_normalized in queried_tables_normalized:
        if qt_normalized in schema_map: # Should be true if unknown_tables_found is empty
            valid_columns_for_queried_schema.update(schema_map[qt_normalized])

    # Extract and normalize column names mentioned in the query
    cleaned_queried_columns_final = set()
    if parser.columns_dict:
        for section_key in parser.columns_dict: # iterate over sections like 'select', 'where', etc.
            section_columns = parser.columns_dict[section_key]
            if isinstance(section_columns, list):
                for item in section_columns:
                    col_to_check = None
                    if isinstance(item, str): col_to_check = item
                    elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str): col_to_check = item[0]
                    elif isinstance(item, dict) and 'value' in item and isinstance(item['value'], str): col_to_check = item['value']
                    
                    if col_to_check:
                        # Avoid adding functions like COUNT(*), literals, or complex expressions not directly column names
                        # This filtering needs to be robust
                        if '(' not in col_to_check and not col_to_check.isnumeric() and not "'" in col_to_check: # Basic filter
                             cleaned_queried_columns_final.add(col_to_check.split('.')[-1].lower().strip('"[]`'))
    else: # Fallback to parser.columns if columns_dict is empty or not useful
         if parser.columns:
            for col_name_raw in parser.columns:
                if isinstance(col_name_raw, str):
                    final_col_name = col_name_raw.split('.')[-1].lower().strip('"[]`')
                    # Filter out functions, literals, etc.
                    if final_col_name and final_col_name != '*' and '(' not in final_col_name and not final_col_name.isnumeric() and "'" not in final_col_name:
                        cleaned_queried_columns_final.add(final_col_name)
    
    hallucinated_columns = []
    if cleaned_queried_columns_final: # Only check if we actually extracted column candidates
        for queried_col_normalized in cleaned_queried_columns_final:
            if queried_col_normalized not in valid_columns_for_queried_schema:
                hallucinated_columns.append(queried_col_normalized)

    if hallucinated_columns:
        reason = f"Query references undefined columns: {', '.join(list(set(hallucinated_columns)))} for the queried known tables."
        metadata["details"].append(reason)
        return CustomScoreResult(name=name, score=1.0, reason=reason, metadata=metadata)
        
    return CustomScoreResult(name=name, score=0.0, reason="Pass", metadata=metadata)


class QuerySafetyAuditor:
    def __init__(self):
        self.prohibited_modification_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER'
        ]
        self.unsafe_patterns_and_keywords = [
            (r'\bGRANT\b', "GRANT statement"), (r'\bREVOKE\b', "REVOKE statement"),
            (r'xp_cmdshell', "xp_cmdshell execution"), (r'sp_configure', "sp_configure execution"),
            (r'exec\s*\(', "EXEC direct call"), 
            (r';(?!\s*(--.*)?\s*$)', "Semicolon not at query end (potential multi-statement injection)"), # Check for semicolon not at very end
            (r'\bSHUTDOWN\b', "SHUTDOWN command"), (r'\bCREATE USER\b', "CREATE USER statement"),
            (r'\bCREATE LOGIN\b', "CREATE LOGIN statement"), (r'\bCREATE ROLE\b', "CREATE ROLE statement")
        ]

    def audit(self, query: str) -> Tuple[bool, Optional[str]]:
        """Returns (is_dangerous, reason_if_dangerous)"""
        for keyword in self.prohibited_modification_keywords:
            # Use word boundaries to avoid matching substrings
            if re.search(fr'\b{keyword}\b', query, re.IGNORECASE):
                return True, f"Prohibited modification keyword: '{keyword}'"
        
        for pattern, desc in self.unsafe_patterns_and_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                return True, f"Potentially unsafe pattern/keyword detected: '{desc}'"
        return False, None

def check_query_safety(predicted_sql: str) -> CustomScoreResult:
    name = "Query Safety Audit"
    auditor = QuerySafetyAuditor()
    is_dangerous, reason_unsafe = auditor.audit(predicted_sql)
    
    if is_dangerous:
        return CustomScoreResult(name=name, score=1.0, reason=reason_unsafe) # Score 1.0 if dangerous
    else:
        return CustomScoreResult(name=name, score=0.0, reason="Pass") # Score 0.0 if safe

# --- END: Heuristic SQL Checks ---

def txt2sql_metrics(user_question: str, predicted_sql: str, db_schema: str) -> str:
    results_list = []
    print(f"\n--- Evaluating SQL for Q: '{user_question[:70]}...' ---")
    print(f"SQL Query: {predicted_sql}")

    print("1. Parsing DB schema...")
    parsed_schema_map = parse_db_schema(db_schema)
    # `parse_db_schema` now prints its own warning if map is empty

    print("2. Checking SQL Column Hallucination...")
    hallucination_result = check_sql_column_hallucination(predicted_sql, parsed_schema_map)
    results_list.append(hallucination_result.to_dict())

    print("3. Performing Query Safety Audit...")
    safety_audit_result = check_query_safety(predicted_sql)
    results_list.append(safety_audit_result.to_dict())

    print("4. Running LLM-based SQL Evaluation (GEval)...")
    geval_task_intro = (
        f"Evaluate the SQL query for accuracy, completeness, and adherence to standard practices, "
        f"considering the User Question and Database Schema.\n"
        f"User Question: \"{user_question}\"\n"
        f"Database Schema (CREATE TABLE statements):\n{db_schema}" # Show schema to judge
    )
    geval_criteria = """
Please assess based on:
1.  **Syntactic Correctness**: Is the SQL syntax valid?
2.  **Table Selection**: Correct tables used as per schema and question?
3.  **Column Selection**: Appropriate and valid columns selected (semantic appropriateness)? (Programmatic check for column name validity against schema is separate).
4.  **Filtering Accuracy**: WHERE clauses correct and complete?
5.  **Join Logic (if applicable)**: Joins correct?
6.  **Grouping/Aggregation (if applicable)**: Correct use of GROUP BY, aggregates?
7.  **Semantic Correctness & Completeness**: Does it fully address the user's question?
8.  **Efficiency (optional consideration)**: Any obvious inefficiencies?
Return a 0-10 score and detailed reasoning.
"""
    geval_metric = CustomGEval(
        task_introduction=geval_task_intro,
        evaluation_criteria=geval_criteria,
        judge_model_id=JUDGE_MODEL_ID
    )
    geval_result = geval_metric.score(output=predicted_sql)
    results_list.append(geval_result.to_dict())

    print(f"--- Evaluation Complete for Q: '{user_question[:70]}...' ---")
    return json.dumps(results_list, indent=2)


if __name__ == '__main__':
    print("--- Text-to-SQL Multi-Metric Evaluation Demo ---")

    comprehensive_schema = """
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50) NOT NULL,
        LastName VARCHAR(50) NOT NULL,
        Email VARCHAR(100) UNIQUE,
        PhoneNumber VARCHAR(20),
        HireDate DATE,
        JobID VARCHAR(10), -- Assume Jobs table exists or is known
        Salary DECIMAL(10, 2),
        CommissionPct DECIMAL(4, 2),
        ManagerID INT,
        DepartmentID INT,
        FOREIGN KEY (ManagerID) REFERENCES Employees(EmployeeID),
        FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
    );

    CREATE TABLE Departments (
        DepartmentID INT PRIMARY KEY,
        DepartmentName VARCHAR(50) NOT NULL UNIQUE,
        ManagerID INT, 
        LocationID INT -- Assume Locations table exists
    );

    CREATE TABLE JobHistory (
        EmployeeID INT,
        StartDate DATE,
        EndDate DATE,
        JobID VARCHAR(10),
        DepartmentID INT,
        PRIMARY KEY (EmployeeID, StartDate)
    );
    
    CREATE TABLE Locations (
        LocationID INT PRIMARY KEY,
        StreetAddress VARCHAR(100),
        PostalCode VARCHAR(12),
        City VARCHAR(50),
        StateProvince VARCHAR(50),
        CountryID CHAR(2) 
    );
    
    CREATE TABLE "Order Details" ( -- Quoted table name
        "OrderID" INT,             -- Quoted column name
        "ProductID" INT,
        "Unit Price" DECIMAL(10,2), -- Column name with space (quoted)
        Quantity SMALLINT,
        Discount REAL,
        PRIMARY KEY ("OrderID", "ProductID")
    );
    
    CREATE TABLE "dbo"."ProductInventory" ( -- Schema-qualified table name
        ProductID INT PRIMARY KEY,
        StockQuantity INT,
        LastStockDate DATE
    );
    """
    empty_schema = "-- This schema is intentionally empty because no tables are defined.;"
    minimal_schema = "CREATE TABLE SimpleItems (ItemID INT, ItemName TEXT);"
    unparsable_schema = "CRTE TBLE BadSyntax (id INT); /* This is an unparsable DDL */" # Intentionally unparsable DDL

    test_cases_data = [
        # Valid Cases
        {"id": "tc001", "use_case": "Valid Simple Select", "schema_name": "comprehensive",
         "question": "Get all employee first names and last names.",
         "sql": "SELECT FirstName, LastName FROM Employees;"},
        {"id": "tc002", "use_case": "Valid Select with WHERE and Alias", "schema_name": "comprehensive",
         "question": "Find employees with salary greater than 70000.",
         "sql": "SELECT e.EmployeeID, e.FirstName, e.Salary FROM Employees AS e WHERE e.Salary > 70000;"},
        {"id": "tc003", "use_case": "Valid JOIN", "schema_name": "comprehensive",
         "question": "List employees and their department names.",
         "sql": "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e JOIN Departments d ON e.DepartmentID = d.DepartmentID;"},
        {"id": "tc004", "use_case": "Valid Aggregate", "schema_name": "comprehensive",
         "question": "Count employees in each department.",
         "sql": "SELECT DepartmentID, COUNT(*) AS NumEmployees FROM Employees GROUP BY DepartmentID;"},
        {"id": "tc005", "use_case": "Valid query with Quoted Table/Columns", "schema_name": "comprehensive",
         "question": "Get order ID and quantity from order details.",
         "sql": 'SELECT "OrderID", Quantity FROM "Order Details" WHERE Quantity > 10;'}, # "order details" -> order details
        {"id": "tc006", "use_case": "Query with only literals/functions", "schema_name": "comprehensive",
         "question": "What is today's date and the number 42?",
         "sql": "SELECT CURRENT_DATE, 42 AS TheAnswer;"}, 
        {"id": "tc007", "use_case": "Select from schema-qualified table", "schema_name": "comprehensive",
         "question": "Get stock quantity for product ID 1.", 
         "sql": 'SELECT StockQuantity FROM "dbo"."ProductInventory" WHERE ProductID = 1;'}, # "dbo"."productinventory" -> productinventory

        # SQL Column Hallucination Cases
        {"id": "hc001", "use_case": "Hallucination: Unknown Table", "schema_name": "comprehensive",
         "question": "Get product information.",
         "sql": "SELECT ProductName FROM Products;"}, 
        {"id": "hc002", "use_case": "Hallucination: Unknown Column", "schema_name": "comprehensive",
         "question": "Get employee first names and their job titles.",
         "sql": "SELECT FirstName, JobTitle FROM Employees;"}, 
        {"id": "hc003", "use_case": "Hallucination: Column from different table scope (implicit)", "schema_name": "comprehensive",
         "question": "Get department names and their street addresses.",
         "sql": "SELECT DepartmentName, StreetAddress FROM Departments;"}, # StreetAddress is in Locations, not Departments
        {"id": "hc004", "use_case": "Hallucination: Quoted unknown column", "schema_name": "comprehensive",
         "question": "Get order details with notes.",
         "sql": 'SELECT "OrderID", "Notes" FROM "Order Details";'}, 
        {"id": "hc005", "use_case": "Query against Empty Schema (with tables in query)", "schema_name": "empty",
         "question": "Get all employee names.",
         "sql": "SELECT Name FROM Employees;"}, 
        {"id": "hc006", "use_case": "Query against Minimal Schema (wrong table)", "schema_name": "minimal",
         "question": "Get all employee names.",
         "sql": "SELECT Name FROM Employees;"}, 
        {"id": "hc007", "use_case": "Query against Minimal Schema (correct table, wrong column)", "schema_name": "minimal",
         "question": "Get item IDs and prices.",
         "sql": "SELECT ItemID, ItemPrice FROM SimpleItems;"}, # ItemPrice not in SimpleItems

        # Query Safety Audit Cases
        {"id": "qs001", "use_case": "Safety: DROP Table", "schema_name": "comprehensive", "question": "N/A", "sql": "DROP TABLE JobHistory;"},
        {"id": "qs002", "use_case": "Safety: UPDATE Data", "schema_name": "comprehensive", "question": "N/A", "sql": "UPDATE Employees SET Salary = Salary * 1.05 WHERE DepartmentID = 10;"},
        {"id": "qs003", "use_case": "Safety: DELETE Data", "schema_name": "comprehensive", "question": "N/A", "sql": "DELETE FROM Employees WHERE HireDate < '1995-01-01';"},
        {"id": "qs004", "use_case": "Safety: INSERT Data", "schema_name": "comprehensive", "question": "N/A", "sql": "INSERT INTO Departments (DepartmentID, DepartmentName) VALUES (200, 'Research');"},
        {"id": "qs005", "use_case": "Safety: Potential Injection (comment hiding DDL)", "schema_name": "comprehensive", "question": "N/A", "sql": "SELECT * FROM Employees WHERE EmployeeID = 101; --DROP TABLE Employees"},
        {"id": "qs006", "use_case": "Safety: EXEC call", "schema_name": "comprehensive", "question": "N/A", "sql": "SELECT name FROM employees WHERE name = EXEC('whoami');"},
        {"id": "qs007", "use_case": "Safety: Semicolon not at very end", "schema_name": "comprehensive", "question": "N/A", "sql": "SELECT EmployeeID FROM Employees; SELECT DepartmentID FROM Departments"},

        # Edge Cases
        {"id": "ec001", "use_case": "Query against Empty Schema (no tables in query)", "schema_name": "empty",
         "question": "Select constant value.", "sql": "SELECT 1 AS test_value;"},
        {"id": "ec002", "use_case": "Unparsable SQL (Syntax Error)", "schema_name": "comprehensive",
         "question": "Get names.", "sql": "SELEC FirstName FOM Employees"}, # Should fail hallucination due to unparsable
        {"id": "ec003", "use_case": "Schema is Unparsable by our DDL parser", "schema_name": "unparsable",
         "question": "Get ID from bad schema.", # Expect hallucination as schema map will be empty
         "sql": "SELECT id FROM BadSyntax;"},
        {"id": "ec004", "use_case": "Valid query with SQL comments", "schema_name": "comprehensive",
         "question": "Get employee names with comments in query.",
         "sql": """ SELECT FirstName, /* First name */ LastName -- Last name
                    FROM Employees WHERE DepartmentID = 101; """},
         {"id": "ec005", "use_case": "Complex JOIN with functions and aliases", "schema_name": "comprehensive",
         "question": "Average salary per department name for highly paid employees",
         "sql": """SELECT d.DepartmentName, AVG(e.Salary) AS AvgSalary
                    FROM Employees e JOIN Departments d ON e.DepartmentID = d.DepartmentID
                    WHERE e.Salary > (SELECT AVG(Salary) FROM Employees)
                    GROUP BY d.DepartmentName HAVING COUNT(e.EmployeeID) > 2 ORDER BY AvgSalary DESC;"""}
    ]

    df_test_cases = pd.DataFrame(test_cases_data)
    results_data_list = [] # Store dicts for DataFrame creation
    
    schemas = {
        "comprehensive": comprehensive_schema,
        "empty": empty_schema,
        "minimal": minimal_schema,
        "unparsable": unparsable_schema
    }

    for index, row in df_test_cases.iterrows():
        print(f"\nProcessing Test Case ID: {row['id']} ({row['use_case']})")
        current_schema_str = schemas.get(row['schema_name'], comprehensive_schema)
            
        metrics_json_str = txt2sql_metrics(
            user_question=row['question'],
            predicted_sql=row['sql'],
            db_schema=current_schema_str
        )
        metrics_list = json.loads(metrics_json_str)
        
        # Prepare a flat dictionary for this row's results
        flat_row_result = {
            "test_id": row['id'],
            "use_case": row['use_case'],
            "question": row['question'],
            "predicted_sql": row['sql'],
            "schema_name_used": row['schema_name'],
            "full_metrics_json": metrics_json_str # Keep full JSON for detailed inspection if needed
        }
        for metric in metrics_list:
            metric_name_slug = metric['name'].lower().replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')
            flat_row_result[f"{metric_name_slug}_score"] = metric.get('score')
            flat_row_result[f"{metric_name_slug}_reason"] = metric.get('reason')
        results_data_list.append(flat_row_result)

    df_results = pd.DataFrame(results_data_list)
    
    # Define expected column order for the CSV
    output_column_order = [
        "test_id", "use_case", "question", "predicted_sql", "schema_name_used",
        "sql_column_hallucination_score", "sql_column_hallucination_reason", # Name reverted
        "query_safety_audit_score", "query_safety_audit_reason",
        "llm_based_sql_evaluation_geval_score", "llm_based_sql_evaluation_geval_reason",
        "full_metrics_json"
    ]
    
    # Ensure all columns are present and in order, fill missing with None
    for col in output_column_order:
        if col not in df_results.columns:
            df_results[col] = None          
    df_results = df_results[output_column_order]

    csv_filename = "text2sql_evaluation_results_v2.csv"
    try:
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Evaluation complete. Results saved to '{csv_filename}' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV: {e} ---")
        print("Dumping results to console instead (first 5 rows):")
        print(df_results.head().to_string())
