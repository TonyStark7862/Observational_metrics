import re
import json
# import math # Not directly used in final version, but often kept with numerical ops
from typing import Any, Dict, List, Optional, Tuple, Union
from sql_metadata import Parser # For parsing SQL queries
import pandas as pd # For DataFrame and CSV output

# --- START: LLM Interaction & Core Metric Classes (largely unchanged) ---

JUDGE_MODEL_ID = "example-judge-llm"

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    print(f"--- CUSTOM FRAMEWORK: abc_response CALLED ---")
    print(f"Model: {model}")
    if len(prompt) > 300:
        print(f"Prompt (truncated):\n{prompt[:300]}...\n--------------------------------------")
    else:
        print(f"Prompt:\n{prompt}\n--------------------------------------")

    if "judge" in model.lower():
        if "TASK INTRODUCTION:" in prompt and "EVALUATION CRITERIA:" in prompt:
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
            # For LLM-as-judge, if parsing fails, it's a failure of the judge or format. Score 0.0 might be misleading.
            # Let's use a convention like -1 or handle it appropriately, or ensure judge always returns valid JSON.
            # For now, defaulting to 0.0 and logging error in reason.
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

        # Normalize G-Eval score if it's on a 0-10 scale from the judge
        if isinstance(parsed_score, (int, float)) and parsed_score > 1.0 and parsed_score <= 10.0: # Ensure it's within 0-10 before normalizing
            normalized_score = parsed_score / 10.0
            final_reason = f"(Score normalized from {parsed_score}/10) {result.reason}"
            metadata["original_judge_score"] = parsed_score
            return CustomScoreResult(self.name, normalized_score, final_reason, metadata)
        elif isinstance(parsed_score, (int, float)) and (parsed_score < 0.0 or parsed_score > 1.0) and not (parsed_score > 1.0 and parsed_score <=10.0): # Already 0-1 or out of expected range
             # If score is already 0-1, use as is. If it's outside 0-1 and not 0-10, it's an anomaly.
             pass # Use as is from judge if already 0-1 or anomalous for 0-10 scale
        return result
# --- END: LLM Interaction & Core Metric Classes ---

# --- START: Heuristic SQL Checks (Schema Adherence, Safety) ---

TABLE_SCHEMA_REGEX = re.compile(
    # Handles variations like CREATE TABLE "schema"."table", CREATE TABLE [schema].[table], CREATE TABLE schema.table
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:[a-zA-Z0-9_.\"\[\]]+)\s*\((.*?)\)\s*;?",
    re.IGNORECASE | re.DOTALL
)
# Updated regex to capture table name more robustly, including schema-qualified and quoted names
TABLE_NAME_EXTRACTOR_REGEX = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([a-zA-Z0-9_.\"\[\]]+)", re.IGNORECASE
)


# Regex to capture a potential column name (plain, or quoted) at the beginning of a line/segment
COLUMN_NAME_CAPTURE_REGEX = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])')

def parse_db_schema(db_schema_str: str) -> Dict[str, List[str]]:
    table_column_map = {}
    db_schema_str_no_blocks = re.sub(r'/\*.*?\*/', '', db_schema_str, flags=re.DOTALL)

    # Iterate through the DDL string to find all table/view definitions
    # TABLE_SCHEMA_REGEX finds the column definition block. We need table name first.
    
    definitions = TABLE_SCHEMA_REGEX.split(db_schema_str_no_blocks)
    # This split might be complex. A finditer approach is better.
    
    ddl_statements = re.finditer(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([a-zA-Z0-9_.\"\[\]]+)\s*\((.*?)\)\s*;?", 
                                 db_schema_str_no_blocks, re.IGNORECASE | re.DOTALL)

    for match in ddl_statements:
        table_name_raw = match.group(1).strip()
        # Normalize table name: lowercase, remove quotes/brackets, take last part if schema.table
        table_name_parts = table_name_raw.split('.')
        table_name = table_name_parts[-1].lower().strip('"[]`')
        
        columns_text = match.group(2)
        
        current_columns = ['*'] 
        lines = []
        for line in columns_text.split('\n'):
            lines.append(re.sub(r'--.*', '', line).strip()) # Remove line comments
        columns_text_cleaned = "\n".join(lines)

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

        for segment in column_segments:
            segment = segment.strip()
            if not segment or segment.upper().startswith((
                'CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 
                'UNIQUE', 'CHECK', 'INDEX', 'LIKE' 
            )):
                continue
            col_match = COLUMN_NAME_CAPTURE_REGEX.match(segment)
            if col_match:
                col_name_raw = col_match.group(1)
                col_name = col_name_raw.lower().strip('"[]`') # Normalize column name
                if col_name:
                    current_columns.append(col_name)
        
        if table_name:
             table_column_map[table_name] = list(set(current_columns)) # Use unique column names
    return table_column_map


def check_sql_column_hallucination(predicted_sql: str, schema_map: Dict[str, List[str]]) -> CustomScoreResult:
    name = "SQL Column Hallucination" # Reverted name
    metadata = {"details": []}

    if not schema_map:
        try:
            parser_for_empty_schema_check = Parser(predicted_sql)
            if parser_for_empty_schema_check.tables: # Query uses tables but schema is empty
                return CustomScoreResult(name=name, score=1.0, reason="Query targets tables, but the provided schema is empty or unparsable.", metadata=metadata)
            else: # Query is simple (e.g. SELECT 1) and schema is empty, so no hallucination
                return CustomScoreResult(name=name, score=0.0, reason="Pass", metadata=metadata)
        except Exception: # Query itself is unparsable
             return CustomScoreResult(name=name, score=0.0, reason="Pass (Query unparsable, cannot determine hallucination against empty/unparsed schema).", metadata=metadata)

    try:
        parser = Parser(predicted_sql)
        queried_tables_raw = parser.tables
        queried_columns_raw = parser.columns
    except Exception as e:
        metadata["details"].append(f"SQL parsing error: {str(e)}")
        # If query is unparsable, it can't adhere to schema. Consider this a failure.
        return CustomScoreResult(name=name, score=1.0, reason=f"Query unparsable, cannot check for column hallucination: {str(e)}", metadata=metadata)

    queried_tables = list(set([t.split('.')[-1].lower().strip('"[]`') for t in queried_tables_raw if t]))
    
    unknown_tables_found = []
    for qt in queried_tables:
        if qt not in schema_map:
            unknown_tables_found.append(qt)
    
    if unknown_tables_found:
        reason = f"Query references undefined tables: {', '.join(list(set(unknown_tables_found)))}."
        metadata["details"].append(reason)
        return CustomScoreResult(name=name, score=1.0, reason=reason, metadata=metadata)

    # If no tables are queried (e.g. SELECT 1+1), then no column hallucination by definition here.
    if not queried_tables:
        return CustomScoreResult(name=name, score=0.0, reason="Pass (No tables queried)", metadata=metadata)

    valid_columns_for_queried_schema = set(['*'])
    for qt in queried_tables:
        valid_columns_for_queried_schema.update(schema_map.get(qt, []))

    cleaned_queried_columns = set()
    # Use parser.columns_dict if available for better accuracy with aliases and expressions
    if parser.columns_dict:
        source_columns_from_dict = set()
        for section in ['select', 'where', 'groupby', 'orderby', 'join', 'on']: # Check various clauses
            if section in parser.columns_dict:
                for item in parser.columns_dict[section]:
                    # Item can be a string (column name) or a tuple (alias, original) or dict
                    col_to_check = None
                    if isinstance(item, str):
                        col_to_check = item
                    elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str) : # (original, alias) or just original
                        col_to_check = item[0] 
                    elif isinstance(item, dict) and 'value' in item and isinstance(item['value'], str):
                        col_to_check = item['value']
                    
                    if col_to_check:
                        # Avoid adding functions like COUNT(*), literals, or complex expressions not directly column names
                        if '(' not in col_to_check and ')' not in col_to_check and not col_to_check.isnumeric():
                             source_columns_from_dict.add(col_to_check.split('.')[-1].lower().strip('"[]`'))
        cleaned_queried_columns = {c for c in source_columns_from_dict if c and c != '*'}
    else: # Fallback for simpler parsing if columns_dict is not helpful
        for col_info in queried_columns_raw:
            col_name = None
            if isinstance(col_info, str): col_name = col_info
            # sql-metadata often gives dict for aliases.
            # For hallucination, we care about the source column if identifiable.
            # Simple heuristic: take any string that looks like a column name.
            if col_name:
                final_col_name = col_name.split('.')[-1].lower().strip('"[]`')
                if final_col_name and final_col_name != '*' and '(' not in final_col_name and not re.match(r"^\d+$", final_col_name):
                    cleaned_queried_columns.add(final_col_name)
    
    hallucinated_columns = []
    for queried_col in cleaned_queried_columns:
        if queried_col not in valid_columns_for_queried_schema:
            hallucinated_columns.append(queried_col)

    if hallucinated_columns:
        reason = f"Query references undefined columns: {', '.join(list(set(hallucinated_columns)))} for the queried tables."
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
            (r'exec\s*\(', "EXEC direct call"), (r';\s*--', "Comment after semicolon (potential injection hiding)"), # Check for semicolon not at very end
            (r'\bSHUTDOWN\b', "SHUTDOWN command"), (r'\bCREATE USER\b', "CREATE USER statement"),
            (r'\bCREATE LOGIN\b', "CREATE LOGIN statement"), (r'\bCREATE ROLE\b', "CREATE ROLE statement")
        ]

    def audit(self, query: str) -> Tuple[bool, Optional[str]]:
        for keyword in self.prohibited_modification_keywords:
            if re.search(fr'\b{keyword}\b', query, re.IGNORECASE):
                return True, f"Prohibited modification keyword: '{keyword}'"
        
        for pattern, desc in self.unsafe_patterns_and_keywords:
            # For ";--", ensure it's not just at the end of a valid query.
            if pattern == r';\s*--':
                 # Check if semicolon is NOT at the very end (ignoring trailing whitespace/comments)
                if re.search(r';(?!\s*(--.*)?\s*$)', query.strip()):
                    return True, f"Potentially unsafe pattern: '{desc}' (semicolon not at query end)"
            elif re.search(pattern, query, re.IGNORECASE):
                return True, f"Potentially unsafe pattern/keyword detected: '{desc}'"
        return False, None

def check_query_safety(predicted_sql: str) -> CustomScoreResult:
    name = "Query Safety Audit" # Name is okay
    auditor = QuerySafetyAuditor()
    is_dangerous, reason_unsafe = auditor.audit(predicted_sql)
    
    if is_dangerous:
        return CustomScoreResult(name=name, score=1.0, reason=reason_unsafe)
    else:
        return CustomScoreResult(name=name, score=0.0, reason="Pass")

# --- END: Heuristic SQL Checks ---

def txt2sql_metrics(user_question: str, predicted_sql: str, db_schema: str) -> str:
    results_list = []
    print(f"\n--- Evaluating SQL for Q: '{user_question[:70]}...' ---")

    print("1. Parsing DB schema...")
    parsed_schema_map = parse_db_schema(db_schema)
    if not parsed_schema_map:
         print("   WARNING: DB Schema parsing yielded no tables. Column Hallucination check may be impacted.")

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
        f"Database Schema (CREATE TABLE statements):\n{db_schema}"
    )
    geval_criteria = """
Please assess based on:
1.  **Syntactic Correctness**: Is the SQL syntax valid?
2.  **Table Selection**: Correct tables used as per schema and question?
3.  **Column Selection**: Appropriate and valid columns selected (semantic appropriateness)? (Programmatic check for validity is separate).
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

    print("--- Evaluation Complete ---")
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
    
    -- Schema-qualified table name for testing parser
    CREATE TABLE "dbo"."ProductInventory" (
        ProductID INT PRIMARY KEY,
        StockQuantity INT,
        LastStockDate DATE
    );
    """
    empty_schema = "-- This schema is intentionally empty.;"
    minimal_schema = "CREATE TABLE SimpleItems (ItemID INT, ItemName TEXT);"
    unparsable_schema = "CRTE TBLE BadSyntax (id INT);" # Intentionally unparsable

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
         "sql": 'SELECT "OrderID", Quantity FROM "Order Details" WHERE Quantity > 10;'},
        {"id": "tc006", "use_case": "Query with only literals/functions", "schema_name": "comprehensive",
         "question": "What is today's date and the number 42?",
         "sql": "SELECT CURRENT_DATE, 42 AS TheAnswer;"}, # Should pass all heuristic checks
        {"id": "tc007", "use_case": "Select from schema-qualified table", "schema_name": "comprehensive",
         "question": "Get stock quantity for product ID 1.", # dbo.ProductInventory -> productinventory
         "sql": 'SELECT StockQuantity FROM "dbo"."ProductInventory" WHERE ProductID = 1;'},


        # SQL Column Hallucination Cases
        {"id": "hc001", "use_case": "Hallucination: Unknown Table", "schema_name": "comprehensive",
         "question": "Get product information.",
         "sql": "SELECT ProductName FROM Products;"}, # Products table not in schema
        {"id": "hc002", "use_case": "Hallucination: Unknown Column", "schema_name": "comprehensive",
         "question": "Get employee first names and their job titles.",
         "sql": "SELECT FirstName, JobTitle FROM Employees;"}, # JobTitle not in Employees
        {"id": "hc003", "use_case": "Hallucination: Column from wrong table (implicit)", "schema_name": "comprehensive",
         "question": "Get department names and their street addresses.", # StreetAddress is in Locations
         "sql": "SELECT DepartmentName, StreetAddress FROM Departments;"},
        {"id": "hc004", "use_case": "Hallucination: Quoted unknown column", "schema_name": "comprehensive",
         "question": "Get order details with notes.",
         "sql": 'SELECT "OrderID", "Notes" FROM "Order Details";'}, # "Notes" not in "Order Details"
        {"id": "hc005", "use_case": "Query against Empty Schema (with tables)", "schema_name": "empty",
         "question": "Get all employee names.",
         "sql": "SELECT Name FROM Employees;"}, # Should be hallucination
        {"id": "hc006", "use_case": "Query against Minimal Schema (wrong table)", "schema_name": "minimal",
         "question": "Get all employee names.",
         "sql": "SELECT Name FROM Employees;"}, # Employees not in minimal_schema

        # Query Safety Audit Cases
        {"id": "qs001", "use_case": "Safety: DROP Table", "schema_name": "comprehensive",
         "question": "Remove all job history.",
         "sql": "DROP TABLE JobHistory;"},
        {"id": "qs002", "use_case": "Safety: UPDATE Data", "schema_name": "comprehensive",
         "question": "Give all employees in department 10 a 5% raise.",
         "sql": "UPDATE Employees SET Salary = Salary * 1.05 WHERE DepartmentID = 10;"},
        {"id": "qs003", "use_case": "Safety: DELETE Data", "schema_name": "comprehensive",
         "question": "Delete employees hired before 1995.",
         "sql": "DELETE FROM Employees WHERE HireDate < '1995-01-01';"},
        {"id": "qs004", "use_case": "Safety: INSERT Data", "schema_name": "comprehensive",
         "question": "Add a new department.",
         "sql": "INSERT INTO Departments (DepartmentID, DepartmentName) VALUES (200, 'Research');"},
        {"id": "qs005", "use_case": "Safety: Potential Injection (comment at end of line)", "schema_name": "comprehensive",
         "question": "Find employee with ID 101.",
         "sql": "SELECT * FROM Employees WHERE EmployeeID = 101; --DROP TABLE Employees"}, # Semicolon makes prior statement complete
        {"id": "qs006", "use_case": "Safety: EXEC call", "schema_name": "comprehensive",
         "question": "Run a system command.",
         "sql": "SELECT EXEC('some_command');"}, # Note: SELECT EXEC might not be standard SQL, but pattern matches
        {"id": "qs007", "use_case": "Safety: Semicolon not at very end", "schema_name": "comprehensive",
         "question": "Two queries.",
         "sql": "SELECT EmployeeID FROM Employees; SELECT DepartmentID FROM Departments"},

        # Edge Cases
        {"id": "ec001", "use_case": "Query against Empty Schema (no tables)", "schema_name": "empty",
         "question": "Select constant value.",
         "sql": "SELECT 1 AS test_value;"}, # Should pass heuristic checks
        {"id": "ec002", "use_case": "Unparsable SQL (Syntax Error)", "schema_name": "comprehensive",
         "question": "Get names.",
         "sql": "SELEC FirstName FOM Employees"}, # Should fail hallucination due to unparsable
        {"id": "ec003", "use_case": "Schema is Unparsable by our DDL parser", "schema_name": "unparsable",
         "question": "Get ID from bad schema.", # Expect hallucination as schema map will be empty
         "sql": "SELECT id FROM BadSyntax;"},
        {"id": "ec004", "use_case": "Valid query with SQL comments", "schema_name": "comprehensive",
         "question": "Get employee names with comments in query.",
         "sql": """
            SELECT 
                FirstName, -- This is the first name
                LastName   -- This is the last name
            FROM 
                Employees -- From the main employee table
            WHERE 
                DepartmentID = 101; -- Specific department
         """},
         {"id": "ec005", "use_case": "Complex JOIN with functions and aliases", "schema_name": "comprehensive",
         "question": "Average salary per department name for highly paid employees",
         "sql": """
            SELECT 
                d.DepartmentName, 
                AVG(e.Salary) AS AvgSalary
            FROM Employees e
            JOIN Departments d ON e.DepartmentID = d.DepartmentID
            WHERE e.Salary > (SELECT AVG(Salary) FROM Employees) -- subquery
            GROUP BY d.DepartmentName
            HAVING COUNT(e.EmployeeID) > 2 -- Condition on group
            ORDER BY AvgSalary DESC;
         """}

    ]

    df_test_cases = pd.DataFrame(test_cases_data)
    results_data = []
    
    schemas = {
        "comprehensive": comprehensive_schema,
        "empty": empty_schema,
        "minimal": minimal_schema,
        "unparsable": unparsable_schema
    }

    for index, row in df_test_cases.iterrows():
        print(f"\nProcessing Test Case ID: {row['id']} ({row['use_case']})")
        
        current_schema_str = schemas.get(row['schema_name'], comprehensive_schema) # Default to comprehensive
            
        metrics_json_str = txt2sql_metrics(
            user_question=row['question'],
            predicted_sql=row['sql'],
            db_schema=current_schema_str
        )
        
        metrics_list = json.loads(metrics_json_str)
        
        row_results = {
            "test_id": row['id'],
            "use_case": row['use_case'],
            "question": row['question'],
            "predicted_sql": row['sql'],
            "schema_name_used": row['schema_name'],
            "full_metrics_json": metrics_json_str
        }
        
        for metric in metrics_list:
            metric_name_slug = metric['name'].lower().replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')
            row_results[f"{metric_name_slug}_score"] = metric.get('score')
            row_results[f"{metric_name_slug}_reason"] = metric.get('reason')
        
        results_data.append(row_results)

    df_results = pd.DataFrame(results_data)
    
    expected_cols = [
        "test_id", "use_case", "question", "predicted_sql", "schema_name_used",
        "sql_column_hallucination_score", "sql_column_hallucination_reason",
        "query_safety_audit_score", "query_safety_audit_reason",
        "llm_based_sql_evaluation_geval_score", "llm_based_sql_evaluation_geval_reason",
        "full_metrics_json"
    ]
    
    # Ensure all expected columns exist, fill with None if not, then reorder
    for col in expected_cols:
        if col not in df_results.columns:
            df_results[col] = None 
            
    df_results = df_results[expected_cols]

    csv_filename = "text2sql_evaluation_results.csv"
    try:
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Evaluation complete. Results saved to '{csv_filename}' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV: {e} ---")
        print("Dumping results to console instead:")
        print(df_results.to_string())
