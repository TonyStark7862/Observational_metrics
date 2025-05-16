import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from sql_metadata import Parser # For parsing SQL queries
import pandas as pd # For DataFrame and CSV output

# --- START: LLM Interaction & Core Metric Classes ---

JUDGE_MODEL_ID = "example-judge-llm"

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    print(f"--- CUSTOM FRAMEWORK: abc_response CALLED ---")
    # Commenting out verbose prints for cleaner debug run, enable if needed for abc_response
    # print(f"Model: {model}")
    # if len(prompt) > 300: print(f"Prompt (truncated):\n{prompt[:300]}...")
    # else: print(f"Prompt:\n{prompt}")

    if "judge" in model.lower():
        if "TASK INTRODUCTION:" in prompt and "EVALUATION CRITERIA:" in prompt:
            raw_score = 7 
            reason_val = "G-Eval Mock: Aspects look reasonable. Minor details could be improved. Overall fair."
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
            reason = str(data.get("reason", "No reason by judge."))
            metadata = {k: v for k, v in data.items() if k not in ["score", "reason"]}
            return score, reason, metadata if metadata else {}
        except Exception as e: return 0.0, f"Err parsing judge response: {e}. Resp: '{judge_response_str}'", {"raw_resp": judge_response_str}
    def score_llm_metric(self, **kwargs) -> CustomScoreResult:
        try: prompt_for_judge = self._format_prompt(**kwargs)
        except KeyError as e: return CustomScoreResult(self.name, 0.0, f"Key err format prompt: {e}", metadata=kwargs)
        judge_response_str, _, _, _ = abc_response(self.judge_model_id, prompt_for_judge)
        final_score, reason, metadata = self._parse_judge_response(judge_response_str)
        return CustomScoreResult(self.name, final_score, reason, metadata)

class CustomGEval(CustomLLMAsJudgeMetric):
    PROMPT_TEMPLATE = """*** TASK: Evaluate SQL. Intro: {task_introduction}. Criteria: {evaluation_criteria}. SQL: {output}. Eval JSON: {{"score": <0-10 score>, "reason": "<reason>"}}"""
    def __init__(self, task_introduction: str, evaluation_criteria: str, judge_model_id: str):
        super().__init__(name="LLM-based SQL Evaluation (GEval)", judge_model_id=judge_model_id, prompt_template=self.PROMPT_TEMPLATE)
        self._task_introduction = task_introduction; self._evaluation_criteria = evaluation_criteria
    def _format_prompt(self, output: str, **kwargs) -> str: return self.prompt_template.format(task_introduction=self._task_introduction, evaluation_criteria=self._evaluation_criteria, output=output)
    def score(self, output: str, **kwargs) -> CustomScoreResult:
        result = super().score_llm_metric(output=output)
        if isinstance(result.score, (int, float)) and result.score > 1.0 and result.score <= 10.0:
            norm_score = result.score / 10.0; reason = f"(Norm from {result.score}/10) {result.reason}"
            meta = result.metadata; meta["original_judge_score"] = result.score
            return CustomScoreResult(self.name, norm_score, reason, meta)
        return result
# --- END: LLM Interaction & Core Metric Classes ---

# --- START: Heuristic SQL Checks ---
DDL_STATEMENT_REGEX = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([a-zA-Z0-9_.\"\[\]\s]+)\s*\((.*?)\)\s*;?", re.IGNORECASE | re.DOTALL)
COLUMN_NAME_CAPTURE_REGEX = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])')

def normalize_identifier(identifier_raw: str, is_table_name=False) -> str:
    if not identifier_raw: return ""
    name_to_process = identifier_raw.split('.')[-1] 
    stripped_name = name_to_process.strip('"[]`')
    replaced_spaces_name = stripped_name.replace(' ', '_')
    return replaced_spaces_name.lower()

def parse_db_schema(db_schema_str: str) -> Dict[str, List[str]]:
    print("\n--- Starting DB Schema Parsing ---")
    table_column_map = {}
    db_schema_str_no_blocks = re.sub(r'/\*.*?\*/', '', db_schema_str, flags=re.DOTALL)
    for ddl_match in DDL_STATEMENT_REGEX.finditer(db_schema_str_no_blocks):
        table_name_raw, columns_text_raw = ddl_match.group(1).strip(), ddl_match.group(2)
        print(f"\n  Processing DDL for raw table string: '{table_name_raw}'")
        table_name_normalized = normalize_identifier(table_name_raw, is_table_name=True)
        if not table_name_normalized: print(f"    WARNING: Null normalized table name from '{table_name_raw}'. Skip."); continue
        print(f"    Normalized Table Name Key: '{table_name_normalized}'")
        current_columns = ['*']
        lines = [re.sub(r'--.*', '', line).strip() for line in columns_text_raw.split('\n')]
        columns_text_cleaned = "\n".join(lines)
        balance, current_segment, column_segments = 0, "", []
        for char in columns_text_cleaned:
            if char == '(': balance += 1
            elif char == ')': balance -=1
            if char == ',' and balance == 0: column_segments.append(current_segment.strip()); current_segment = ""
            else: current_segment += char
        column_segments.append(current_segment.strip())
        extracted_cols_norm = []
        for segment in column_segments:
            segment = segment.strip()
            if not segment or segment.upper().startswith(('CONSTRAINT','PRIMARY KEY','FOREIGN KEY','UNIQUE','CHECK','INDEX','LIKE')): continue
            col_match = COLUMN_NAME_CAPTURE_REGEX.match(segment)
            if col_match:
                col_name_norm = normalize_identifier(col_match.group(1))
                if col_name_norm: extracted_cols_norm.append(col_name_norm)
        current_columns.extend(list(set(extracted_cols_norm)))
        table_column_map[table_name_normalized] = list(set(current_columns))
        print(f"    Extracted & Normalized Columns for '{table_name_normalized}': {table_column_map[table_name_normalized]}")
    if not table_column_map: print("  WARNING: Schema parsing resulted in an empty map.")
    print("--- DB Schema Parsing Complete ---")
    return table_column_map

def check_sql_column_hallucination(predicted_sql: str, schema_map: Dict[str, List[str]]) -> CustomScoreResult:
    name = "SQL Column Hallucination"
    metadata = {"details": []}
    print(f"\n  --- Checking Column Hallucination for SQL: {predicted_sql[:100]}...")
    # print(f"    Using Schema Map (first 3 tables): { {k: schema_map[k] for k in list(schema_map)[:3]} }...") # Can be verbose

    if not schema_map:
        try:
            p_empty = Parser(predicted_sql)
            if p_empty.tables: reason = "Query targets tables, but DDL schema is empty/unparsable." ; score = 1.0
            else: reason = "Pass (Query has no tables, schema map empty)"; score = 0.0
        except Exception as e: reason = f"Pass (Query unparsable: {e}, vs empty schema)"; score = 0.0
        metadata["details"].append(reason); print(f"    Hallucination: {reason}"); return CustomScoreResult(name, score, reason, metadata)

    try:
        parser = Parser(predicted_sql)
        queried_tables_raw = parser.tables or []
        cte_names_raw = parser.with_names or []
    except Exception as e:
        reason = f"Query unparsable: {e}"; metadata["details"].append(reason)
        print(f"    Hallucination: FAIL ({reason})"); return CustomScoreResult(name, 1.0, reason, metadata)

    normalized_cte_names = set(normalize_identifier(n, is_table_name=True) for n in cte_names_raw)
    print(f"    Normalized CTEs in Query: {normalized_cte_names}")
    
    queried_table_tokens_normalized = set(normalize_identifier(t, is_table_name=True) for t in queried_tables_raw if t)
    print(f"    All Normalized Table-Like Tokens from parser.tables: {queried_table_tokens_normalized}")

    base_tables_to_check_in_schema = [t for t in queried_table_tokens_normalized if t not in normalized_cte_names]
    print(f"    Base Tables to check against DDL: {base_tables_to_check_in_schema}")
    
    unknown_tables_found = [t for t in base_tables_to_check_in_schema if t not in schema_map]
    if unknown_tables_found:
        reason = f"Query references undefined base tables: {', '.join(unknown_tables_found)}."
        metadata["details"].append(reason); print(f"    Hallucination: FAIL ({reason})"); return CustomScoreResult(name, 1.0, reason, metadata)

    if not base_tables_to_check_in_schema and not normalized_cte_names: # e.g. SELECT 1
        if not (parser.columns or parser.columns_dict): # Truly simple select without column-like tokens
             print("    Hallucination: PASS (No base tables or CTEs queried, simple select)"); return CustomScoreResult(name, 0.0, "Pass (No base tables or CTEs queried)", metadata)

    valid_cols_from_queried_base_ddl = set(['*'])
    for bt_norm in base_tables_to_check_in_schema: # These are confirmed to be in schema_map
        valid_cols_from_queried_base_ddl.update(schema_map.get(bt_norm, []))
    # print(f"    Expected Valid Cols (from DDL for queried base tables): {list(valid_cols_from_queried_base_ddl)[:25]}...")

    cols_to_validate_against_ddl = set()
    # Use columns_dict for better structured info
    if parser.columns_dict:
        for section_key, section_items in parser.columns_dict.items():
            if isinstance(section_items, list):
                for item in section_items:
                    source_name_raw = None
                    if isinstance(item, str): source_name_raw = item
                    elif isinstance(item, tuple) and item: source_name_raw = item[0] # original from (original, alias)
                    elif isinstance(item, dict) and 'value' in item: source_name_raw = item['value'] # value from {'value': orig, 'name': alias}
                    
                    if source_name_raw:
                        # Filter functions/literals - very basic
                        if '(' in source_name_raw or source_name_raw.isnumeric() or ("'" in source_name_raw and source_name_raw.count("'") % 2 == 0):
                            continue
                        
                        parts = source_name_raw.split('.')
                        col_part_norm = normalize_identifier(parts[-1])
                        qualifier_norm = normalize_identifier(parts[0], True) if len(parts) > 1 else None
                        
                        if qualifier_norm and qualifier_norm in normalized_cte_names:
                            # print(f"    Skipping DDL validation for '{col_part_norm}' from CTE '{qualifier_norm}'")
                            continue # Column from a CTE, don't validate against base DDL map
                        
                        if col_part_norm and col_part_norm != '*':
                            cols_to_validate_against_ddl.add(col_part_norm)
    # Fallback or supplement with parser.columns (less structured)
    if not cols_to_validate_against_ddl and parser.columns:
        for col_raw in parser.columns:
            if isinstance(col_raw, str):
                if '(' in col_raw or col_raw.isnumeric() or ("'" in col_raw and col_raw.count("'") % 2 == 0): continue
                # Here, it's hard to know if an unqualified col_raw comes from a CTE or base table if query has both.
                # This is a simplification: if it wasn't picked up via columns_dict with clear sourcing,
                # and it's unqualified, it will be checked against all valid columns from base tables.
                # This might lead to false positives if an unqualified CTE output column has a conflicting name.
                col_norm = normalize_identifier(col_raw.split('.')[-1])
                if col_norm and col_norm != '*':
                    cols_to_validate_against_ddl.add(col_norm)


    print(f"    Normalized Columns Extracted from SQL to validate against DDL: {cols_to_validate_against_ddl}")
    
    hallucinated_cols = [q_col for q_col in cols_to_validate_against_ddl if q_col not in valid_cols_from_queried_base_ddl]
    if hallucinated_cols:
        reason = f"Query references undefined columns: {', '.join(hallucinated_cols)} for the queried known base tables."
        metadata["details"].append(reason); print(f"    Hallucination: FAIL ({reason})"); return CustomScoreResult(name,1.0,reason,metadata)
    
    print("    Hallucination: PASS"); return CustomScoreResult(name, 0.0, "Pass", metadata)

class QuerySafetyAuditor:
    def __init__(self):
        self.keywords = ['DROP','DELETE','TRUNCATE','UPDATE','INSERT','ALTER']
        self.patterns = [(r'\bGRANT\b',"GRANT"),(r'\bREVOKE\b',"REVOKE"),(r'xp_cmdshell',"xp_cmdshell"), (r'sp_configure',"sp_configure"), (r'exec\s*\(',"EXEC()"), (r';(?!\s*(--.*)?\s*$)','Mid-query Semicolon'), (r'\bSHUTDOWN\b',"SHUTDOWN")]
    def audit(self, q: str) -> Tuple[bool, Optional[str]]:
        for kw in self.keywords:
            if re.search(fr'\b{kw}\b', q, re.I): return True, f"Prohibited keyword: '{kw}'"
        for p, d in self.patterns:
            if re.search(p, q, re.I): return True, f"Unsafe pattern: '{d}'"
        return False, None

def check_query_safety(sql: str) -> CustomScoreResult:
    auditor = QuerySafetyAuditor()
    is_dang, reason = auditor.audit(sql)
    return CustomScoreResult("Query Safety Audit", 1.0 if is_dang else 0.0, reason if is_dang else "Pass")

# --- END: Heuristic SQL Checks ---

def txt2sql_metrics(user_question: str, predicted_sql: str, db_schema: str) -> str:
    results = []
    print(f"\n>>> Evaluating Q: '{user_question[:50]}' SQL: {predicted_sql[:60].replace('\n',' ')}...")
    schema_map = parse_db_schema(db_schema)
    results.append(check_sql_column_hallucination(predicted_sql, schema_map).to_dict())
    results.append(check_query_safety(predicted_sql).to_dict())
    # print("  Running GEval...") # GEval is less verbose now
    geval_intro = (f"Q:\"{user_question}\"\nSchema:\n{db_schema}")
    geval_crit = "Crit:Syntax,TblSel,ColSel,Filter,Join,Agg,Semantic,Eff. Score 0-10."
    geval = CustomGEval(geval_intro, geval_crit, JUDGE_MODEL_ID)
    results.append(geval.score(output=predicted_sql).to_dict())
    # print(f"<<< Eval Complete for Q: '{user_question[:50]}'")
    return json.dumps(results, indent=2)

if __name__ == '__main__':
    print("--- Text-to-SQL Multi-Metric Evaluation Demo ---")
    comprehensive_schema = """CREATE TABLE Employees (EmployeeID INT PRIMARY KEY, FirstName VARCHAR(50), LastName VARCHAR(50), Email VARCHAR(100), PhoneNumber VARCHAR(20), HireDate DATE, JobID VARCHAR(10), Salary DECIMAL(10, 2), CommissionPct DECIMAL(4, 2), ManagerID INT, DepartmentID INT); CREATE TABLE Departments (DepartmentID INT PRIMARY KEY, DepartmentName VARCHAR(50) NOT NULL UNIQUE, ManagerID INT, LocationID INT); CREATE TABLE JobHistory (EmployeeID INT, StartDate DATE, EndDate DATE, JobID VARCHAR(10), DepartmentID INT, PRIMARY KEY (EmployeeID, StartDate)); CREATE TABLE Locations (LocationID INT PRIMARY KEY, StreetAddress VARCHAR(100), PostalCode VARCHAR(12), City VARCHAR(50), StateProvince VARCHAR(50), CountryID CHAR(2)); CREATE TABLE "Order Details" ("OrderID" INT, "ProductID" INT, "Unit Price" DECIMAL(10,2), Quantity SMALLINT, Discount REAL, PRIMARY KEY ("OrderID", "ProductID")); CREATE TABLE "dbo"."ProductInventory" (ProductID INT PRIMARY KEY, StockQuantity INT, LastStockDate DATE);"""
    empty_schema = "-- No tables defined;"
    minimal_schema = "CREATE TABLE SimpleItems (ItemID INT, ItemName TEXT, item_cost DECIMAL(5,2)); CREATE TABLE AnotherMinimal (id INT);"
    cte_schema = "CREATE TABLE Sales (Region VARCHAR(50), Amount INT, SalesPersonID INT); CREATE TABLE SalesPeople (SPID INT, SPName VARCHAR(50));" # For CTE test
    unparsable_schema = "CRTE TBLE BadSyntax (id INT);"

    test_cases_data = [
        {"id": "tc001", "uc": "Valid Simple", "sch": "comprehensive", "q": "Names.", "sql": "SELECT FirstName, LastName FROM Employees;"},
        {"id": "tc002", "uc": "Valid Alias", "sch": "comprehensive", "q": "Salary.", "sql": "SELECT e.FirstName, e.Salary AS EmpSalary FROM Employees AS e WHERE e.Salary > 10000;"},
        {"id": "tc003", "uc": "Valid JOIN", "sch": "comprehensive", "q": "DeptNames.", "sql": "SELECT e.FirstName, d.DepartmentName FROM Employees e JOIN Departments d ON e.DepartmentID = d.DepartmentID;"},
        {"id": "tc005", "uc": "Quoted Tbl/Col", "sch": "comprehensive", "q": "Order Qty.", "sql": 'SELECT "OrderID", Quantity FROM "Order Details" WHERE Quantity > 5;'},
        {"id": "tc007", "uc": "Schema-Qual Tbl", "sch": "comprehensive", "q": "Inventory.", "sql": 'SELECT StockQuantity FROM "dbo"."ProductInventory" WHERE ProductID = 1;'},
        {"id": "hc001", "uc": "H: Unk Tbl", "sch": "comprehensive", "q": "Products?", "sql": "SELECT ProductName FROM Products;"},
        {"id": "hc002", "uc": "H: Unk Col 'JobTitle'", "sch": "comprehensive", "q": "Titles?", "sql": "SELECT FirstName, JobTitle FROM Employees;"},
        {"id": "hc003", "uc": "H: Col 'StreetAddress' from wrong scope", "sch": "comprehensive", "q": "Dept Addr?", "sql": "SELECT DepartmentName, StreetAddress FROM Departments;"},
        {"id": "hc004", "uc": "H: Quoted Unk Col 'Notes'", "sch": "comprehensive", "q": "Order Notes?", "sql": 'SELECT "OrderID", "Notes" FROM "Order Details";'},
        {"id": "hc005", "uc": "H: Query Empty Schema (SQL has tables)", "sch": "empty", "q": "Emps?", "sql": "SELECT Name FROM Employees;"},
        {"id": "hc007", "uc": "H: Minimal Sch (correct tbl, wrong col 'ItemPrice')", "sch": "minimal", "q": "Item Prices?", "sql": "SELECT ItemID, ItemPrice FROM SimpleItems;"},
        {"id": "qs001", "uc": "Safe: DROP", "sch": "comprehensive", "q": "N/A", "sql": "DROP TABLE JobHistory;"},
        {"id": "qs005", "uc": "Safe: Injection attempt", "sch": "comprehensive", "q": "N/A", "sql": "SELECT * FROM Employees WHERE EmployeeID = 101; --DROP TABLE Employees"},
        {"id": "ec001", "uc": "Query Empty Sch (SQL no tables)", "sch": "empty", "q": "Constant?", "sql": "SELECT 1;"},
        {"id": "ec002", "uc": "Unparsable SQL", "sch": "comprehensive", "q": "N/A", "sql": "SELEC Name FOM Employees"},
        {"id": "ec003", "uc": "Unparsable Schema", "sch": "unparsable", "q": "ID?", "sql": "SELECT id FROM BadSyntax;"},
        {"id": "cte001", "uc": "Valid CTE", "sch": "cte_schema", "q": "CTE Sales", "sql": "WITH RegionalSales AS (SELECT Region, SUM(Amount) AS TotalSales FROM Sales GROUP BY Region) SELECT Region, TotalSales FROM RegionalSales WHERE TotalSales > 100;"},
        {"id": "cte002", "uc": "CTE with Base Table Join", "sch": "cte_schema", "q": "CTE SalesPerson", "sql": "WITH RPSales AS (SELECT SalesPersonID, SUM(Amount) AS TotalAmt FROM Sales GROUP BY SalesPersonID) SELECT sp.SPName, rs.TotalAmt FROM SalesPeople sp JOIN RPSales rs ON sp.SPID = rs.SalesPersonID;"},
        {"id": "cte003", "uc": "CTE hallucinated base col", "sch": "cte_schema", "q": "CTE bad base col", "sql": "WITH BadCTE AS (SELECT NonExistentColumn FROM Sales) SELECT * FROM BadCTE;"}, # NonExistentColumn should be caught if it's from Sales
        {"id": "cte004", "uc": "CTE select unk col FROM CTE", "sch": "cte_schema", "q": "CTE bad cte col", "sql": "WITH GoodCTE AS (SELECT Region AS MyRegion FROM Sales) SELECT MyRegion, BogusCol FROM GoodCTE;"}, # BogusCol should not be validated against DDL by this heuristic
    ]
    df_test_cases = pd.DataFrame(test_cases_data)
    results_data_list = []
    schemas = {"comprehensive": comprehensive_schema, "empty": empty_schema, "minimal": minimal_schema, "unparsable": unparsable_schema, "cte_schema": cte_schema}

    for idx, row in df_test_cases.iterrows():
        current_schema_str = schemas.get(row['sch'], comprehensive_schema)
        metrics_json_str = txt2sql_metrics(row['q'], row['sql'], current_schema_str)
        metrics_list = json.loads(metrics_json_str)
        flat_row = {"test_id": row['id'], "use_case": row['uc'], "question": row['q'], "predicted_sql": row['sql'], "schema_name_used": row['sch'], "full_metrics_json": metrics_json_str}
        for metric in metrics_list:
            slug = metric['name'].lower().replace('(','').replace(')','').replace(' ', '_').replace('-', '_')
            flat_row[f"{slug}_score"] = metric.get('score'); flat_row[f"{slug}_reason"] = metric.get('reason')
        results_data_list.append(flat_row)

    df_results = pd.DataFrame(results_data_list)
    cols_order = ["test_id","use_case","question","predicted_sql","schema_name_used", "sql_column_hallucination_score","sql_column_hallucination_reason", "query_safety_audit_score","query_safety_audit_reason", "llm_based_sql_evaluation_geval_score","llm_based_sql_evaluation_geval_reason", "full_metrics_json"]
    for col in cols_order:
        if col not in df_results.columns: df_results[col] = None
    df_results = df_results[cols_order]
    csv_filename = "text2sql_evaluation_results_v4_cte_debug.csv"
    try:
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Evaluation complete. Results saved to '{csv_filename}' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV: {e} ---"); print("Dumping results (first 5 rows):\n", df_results.head().to_string())
