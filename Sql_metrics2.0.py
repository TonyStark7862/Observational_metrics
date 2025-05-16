import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from sql_metadata import Parser # For parsing SQL queries
import pandas as pd # For DataFrame and CSV output

# --- START: LLM Interaction & Core Metric Classes ---

JUDGE_MODEL_ID = "example-judge-llm" # Replace with your actual judge model ID

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    # This is a MOCK function. Replace with your actual LLM API call.
    print(f"--- MOCK LLM_RESPONSE CALLED (model: {model}) ---")
    # For brevity in logs, we won't print the full prompt here, but it's available.
    # print(f"Prompt: {prompt[:300]}...") 

    response_text = ""
    if "judge" in model.lower():
        if "TASK INTRODUCTION:" in prompt and "EVALUATION CRITERIA:" in prompt:
            # Simulate G-Eval response
            # Try to make mock reasoning a bit more dynamic based on SQL for demo
            sql_output_for_geval = "SQL not easily extracted from mock prompt"
            try:
                sql_output_match = re.search(r"LLM OUTPUT TO EVALUATE:\s*([\s\S]+?)\s*\*\*\* YOUR EVALUATION:", prompt, re.IGNORECASE | re.DOTALL)
                if sql_output_match:
                    sql_output_for_geval = sql_output_match.group(1).strip()
            except Exception:
                pass

            if "error" in sql_output_for_geval.lower() or "selec" in sql_output_for_geval.lower(): # simple check for bad SQL
                raw_score = 2
                reason_val = "G-Eval Mock: Detected syntax issues or significant errors in SQL."
            elif "drop" in sql_output_for_geval.lower() or "update" in sql_output_for_geval.lower():
                raw_score = 1
                reason_val = "G-Eval Mock: Query contains modification statements, marked low for typical analytical task."
            else:
                raw_score = 8 
                reason_val = f"G-Eval Mock: SQL '{sql_output_for_geval[:30]}...' seems plausible. Semantic fit looks reasonable. No obvious major flaws."
            response_text = json.dumps({"score": raw_score, "reason": reason_val})
        else: 
            response_text = json.dumps({"score": 0.5, "reason": "Neutral assessment from generic judge mock."})
    else: 
        response_text = f"Mock response from {model} for prompt: {prompt[:30]}..."
    
    return response_text, 0.5, len(prompt.split()), len(response_text.split())


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
    def _parse_judge_response(self, judge_response_str: str) -> Tuple[float, str, Dict[str, Any]]: # Ensure metadata is always a dict
        try:
            data = json.loads(judge_response_str); score = float(data.get("score", 0.0))
            reason = str(data.get("reason", "No reason by judge."))
            metadata = {k: v for k, v in data.items() if k not in ["score", "reason"]}
            return score, reason, metadata
        except Exception as e: return 0.0, f"Err parsing judge response: {e}. Resp: '{judge_response_str}'", {"raw_resp": judge_response_str}
    def score_llm_metric(self, **kwargs) -> CustomScoreResult:
        try: prompt_for_judge = self._format_prompt(**kwargs)
        except KeyError as e: return CustomScoreResult(self.name, 0.0, f"Key err format prompt: {e}", metadata=kwargs)
        judge_response_str, _, _, _ = abc_response(self.judge_model_id, prompt_for_judge)
        final_score, reason, metadata = self._parse_judge_response(judge_response_str)
        return CustomScoreResult(self.name, final_score, reason, metadata)

class CustomGEval(CustomLLMAsJudgeMetric):
    PROMPT_TEMPLATE = """*** TASK INTRODUCTION:\n{task_introduction}\n\n*** EVALUATION CRITERIA:\n{evaluation_criteria}\n\n*** LLM OUTPUT TO EVALUATE:\n{output}\n\n*** YOUR EVALUATION (Return JSON with "score" (0-10) and "reason"):"""
    def __init__(self, task_introduction: str, evaluation_criteria: str, judge_model_id: str):
        super().__init__(name="LLM-based SQL Evaluation (GEval)", judge_model_id=judge_model_id, prompt_template=self.PROMPT_TEMPLATE)
        self._task_introduction = task_introduction; self._evaluation_criteria = evaluation_criteria
    def _format_prompt(self, output: str, **kwargs) -> str: return self.prompt_template.format(task_introduction=self._task_introduction, evaluation_criteria=self._evaluation_criteria, output=output)
    def score(self, output: str, **kwargs) -> CustomScoreResult:
        result = super().score_llm_metric(output=output)
        metadata = result.metadata if result.metadata is not None else {} # Ensure metadata is a dict
        if isinstance(result.score, (int, float)) and result.score > 1.0 and result.score <= 10.0:
            norm_score = result.score / 10.0; reason = f"(Norm from {result.score}/10) {result.reason}"
            metadata["original_judge_score"] = result.score
            return CustomScoreResult(self.name, norm_score, reason, metadata)
        return result
# --- END: LLM Interaction & Core Metric Classes ---

# --- START: Heuristic SQL Checks ---
DDL_STATEMENT_REGEX = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([a-zA-Z0-9_.\"\[\]\s]+)\s*\((.*?)\)\s*;?", re.IGNORECASE | re.DOTALL)
COLUMN_NAME_CAPTURE_REGEX = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])') # Handles various quoting

def normalize_identifier(identifier_raw: str, is_table_name=False) -> str:
    """Normalizes SQL identifiers:
    1. For tables, takes last part of schema.table.
    2. Strips standard SQL quotes.
    3. Replaces spaces (from within quotes) with underscores.
    4. Converts to lowercase.
    """
    if not identifier_raw: return ""
    name_to_process = identifier_raw
    if is_table_name:
        name_to_process = identifier_raw.split('.')[-1] 
    
    stripped_name = name_to_process.strip('"[]`') # Strip common quotes
    replaced_spaces_name = stripped_name.replace(' ', '_') # For names like "Order Details"
    return replaced_spaces_name.lower()

def parse_db_schema(db_schema_str: str) -> Dict[str, List[str]]:
    print("\n--- DB Schema Parsing ---")
    table_column_map = {}
    db_schema_str_no_blocks = re.sub(r'/\*.*?\*/', '', db_schema_str, flags=re.DOTALL)

    for ddl_match in DDL_STATEMENT_REGEX.finditer(db_schema_str_no_blocks):
        table_name_raw, columns_text_raw = ddl_match.group(1).strip(), ddl_match.group(2)
        print(f"\n  Found DDL for raw table string: '{table_name_raw}'")
        
        table_name_normalized = normalize_identifier(table_name_raw, is_table_name=True)
        if not table_name_normalized: 
            print(f"    WARNING: Could not derive normalized table name from '{table_name_raw}'. Skipping."); continue
        print(f"    Normalized Table Name (map key): '{table_name_normalized}'")

        current_columns_normalized = ['*'] 
        lines = [re.sub(r'--.*', '', line).strip() for line in columns_text_raw.split('\n')]
        columns_text_cleaned = "\n".join(lines)
        balance, current_segment, column_segments = 0, "", []
        for char in columns_text_cleaned: # Comma splitting with parenthesis balancing
            if char == '(': balance += 1
            elif char == ')': balance -=1
            if char == ',' and balance == 0: column_segments.append(current_segment.strip()); current_segment = ""
            else: current_segment += char
        column_segments.append(current_segment.strip())
        
        for segment in column_segments:
            segment = segment.strip()
            if not segment or segment.upper().startswith(('CONSTRAINT','PRIMARY KEY','FOREIGN KEY','UNIQUE','CHECK','INDEX','LIKE')): continue
            col_match = COLUMN_NAME_CAPTURE_REGEX.match(segment)
            if col_match:
                col_name_norm = normalize_identifier(col_match.group(1)) # Use consistent normalization
                if col_name_norm: current_columns_normalized.append(col_name_norm)
        
        table_column_map[table_name_normalized] = list(set(current_columns_normalized))
        print(f"    Normalized Columns for '{table_name_normalized}': {table_column_map[table_name_normalized]}")
    
    if not table_column_map: print("  WARNING: Schema parsing resulted in an empty map (no tables found).")
    print("--- DB Schema Parsing Complete ---")
    return table_column_map

def check_sql_column_hallucination(predicted_sql: str, schema_map: Dict[str, List[str]]) -> CustomScoreResult:
    name = "SQL Column Hallucination"
    metadata = {
        "parsed_ddl_tables": sorted(list(schema_map.keys())),
        "parsed_ddl_columns_map_json": json.dumps({k: sorted(v) for k, v in schema_map.items()}), # For CSV output
        "sql_extracted_base_tables": [],
        "sql_extracted_columns_for_ddl_validation": []
    }
    print(f"\n  --- Checking Column Hallucination for SQL: {predicted_sql[:100].replace('\n',' ')}...")
    # print(f"    Using Schema Map (first 3): { {k: schema_map[k] for k in list(schema_map)[:3]} }")

    if not schema_map:
        try: p = Parser(predicted_sql)
        except Exception: p = None # SQL also unparsable
        if p and p.tables: reason = "Query targets tables, but DDL schema is effectively empty/unparsable."; score = 1.0
        else: reason = "Pass (Query has no tables or is unparsable, and schema map is empty)"; score = 0.0
        metadata["details"] = [reason]; print(f"    Result: {reason}"); return CustomScoreResult(name, score, reason, metadata)

    try:
        parser = Parser(predicted_sql)
        queried_tables_raw = parser.tables or []
        cte_names_raw = parser.with_names or []
    except Exception as e:
        reason = f"Query unparsable: {str(e)}"; metadata["details"] = [reason]
        print(f"    Result: FAIL ({reason})"); return CustomScoreResult(name, 1.0, reason, metadata)

    normalized_cte_names = set(normalize_identifier(n, True) for n in cte_names_raw)
    print(f"    Normalized CTEs in Query: {normalized_cte_names if normalized_cte_names else 'None'}")
    
    all_queried_table_tokens_normalized = set(normalize_identifier(t, True) for t in queried_tables_raw if t)
    # print(f"    All Normalized Table-Like Tokens from parser.tables: {all_queried_table_tokens_normalized}")

    base_tables_to_check_in_schema = sorted([t for t in all_queried_table_tokens_normalized if t not in normalized_cte_names])
    metadata["sql_extracted_base_tables"] = base_tables_to_check_in_schema
    print(f"    Base Tables from SQL to check against DDL: {base_tables_to_check_in_schema if base_tables_to_check_in_schema else 'None'}")
    
    unknown_tables_found = [t for t in base_tables_to_check_in_schema if t not in schema_map]
    if unknown_tables_found:
        reason = f"Query references undefined base tables: {', '.join(sorted(list(set(unknown_tables_found))))}."
        metadata["details"].append(reason); print(f"    Result: FAIL ({reason})"); return CustomScoreResult(name, 1.0, reason, metadata)

    if not base_tables_to_check_in_schema and not normalized_cte_names:
        if not (parser.columns or parser.columns_dict):
             print("    Result: PASS (No base tables or CTEs, and no columns found in query)"); return CustomScoreResult(name,0.0,"Pass (No base tables or CTEs queried, simple select)",metadata)

    valid_cols_from_ddl_for_queried_bases = set(['*'])
    for bt_norm in base_tables_to_check_in_schema:
        valid_cols_from_ddl_for_queried_bases.update(schema_map.get(bt_norm, []))
    # print(f"    Expected Valid Cols (from DDL for queried base tables): {sorted(list(valid_cols_from_ddl_for_queried_bases))[:15]}...")

    cols_to_validate_against_ddl = set()
    if parser.columns_dict: # Preferred source for columns
        for section_items in parser.columns_dict.values(): # Iterate over values (lists of columns/aliases)
            if isinstance(section_items, list):
                for item in section_items:
                    source_name_raw = None # This is the name as seen by sql-metadata, potentially qualified
                    if isinstance(item, str): source_name_raw = item
                    elif isinstance(item, tuple) and item: source_name_raw = item[0] # original from (original, alias)
                    elif isinstance(item, dict) and 'value' in item: source_name_raw = item['value']
                    
                    if source_name_raw:
                        if '(' in source_name_raw or source_name_raw.isnumeric() or ("'" in source_name_raw and source_name_raw.count("'") % 2 == 0): continue # Skip functions/literals
                        
                        parts = source_name_raw.split('.')
                        col_part_norm = normalize_identifier(parts[-1])
                        qualifier_norm = normalize_identifier(parts[0], True) if len(parts) > 1 else None
                        
                        if qualifier_norm and qualifier_norm in normalized_cte_names: continue # Skip columns from CTEs for DDL validation
                        
                        if col_part_norm and col_part_norm != '*': cols_to_validate_against_ddl.add(col_part_norm)
    
    # Fallback or supplement with parser.columns (less structured, be cautious)
    # This part can add unqualified columns, which are then checked against the pool of all valid columns from queried base tables.
    if parser.columns: # parser.columns is a flat list
        for col_raw in parser.columns:
            if isinstance(col_raw, str):
                if '(' in col_raw or col_raw.isnumeric() or ("'" in col_raw and col_raw.count("'") % 2 == 0): continue
                # If already processed via columns_dict's more structured info, sql-metadata might list it here too.
                # The main risk is if col_raw is an alias name not caught by columns_dict's structure AND not a CTE qualifier.
                
                parts = col_raw.split('.')
                col_part_norm = normalize_identifier(parts[-1])
                qualifier_norm = normalize_identifier(parts[0], True) if len(parts) > 1 else None

                is_part_of_known_alias = False # check if this col_raw is an alias name itself from SELECT
                if parser.columns_aliases and col_raw in parser.columns_aliases: # col_raw is an alias like "EmpSalary"
                    is_part_of_known_alias = True

                if qualifier_norm and qualifier_norm in normalized_cte_names: continue # Skip if qualified by CTE
                if is_part_of_known_alias : continue # Don't validate the alias name itself

                if col_part_norm and col_part_norm != '*':
                    cols_to_validate_against_ddl.add(col_part_norm)
    
    metadata["sql_extracted_columns_for_ddl_validation"] = sorted(list(cols_to_validate_against_ddl))
    print(f"    Normalized Columns from SQL to validate against DDL: {metadata['sql_extracted_columns_for_ddl_validation'] if metadata['sql_extracted_columns_for_ddl_validation'] else 'None'}")
    
    hallucinated_cols = [q_col for q_col in cols_to_validate_against_ddl if q_col not in valid_cols_from_ddl_for_queried_bases]
    if hallucinated_cols:
        reason = f"Query references undefined columns: {', '.join(sorted(list(set(hallucinated_cols))))} for the queried known base tables."
        metadata["details"].append(reason); print(f"    Result: FAIL ({reason})"); return CustomScoreResult(name,1.0,reason,metadata)
    
    print("    Result: PASS"); return CustomScoreResult(name, 0.0, "Pass", metadata)

class QuerySafetyAuditor: # Same as before
    def __init__(self):
        self.keywords=['DROP','DELETE','TRUNCATE','UPDATE','INSERT','ALTER']
        self.patterns=[(r'\bGRANT\b',"GRANT"),(r'\bREVOKE\b',"REVOKE"),(r'xp_cmdshell',"xp_cmdshell"),(r'sp_configure',"sp_configure"),(r'exec\s*\(',"EXEC()"),(r';(?!\s*(--.*)?\s*$)','Mid-query Semicolon'),(r'\bSHUTDOWN\b',"SHUTDOWN")]
    def audit(self, q: str) -> Tuple[bool, Optional[str]]:
        for kw in self.keywords:
            if re.search(fr'\b{kw}\b', q, re.I): return True, f"Prohibited keyword: '{kw}'"
        for p, d in self.patterns:
            if re.search(p, q, re.I): return True, f"Unsafe pattern: '{d}'"
        return False, None

def check_query_safety(sql: str) -> CustomScoreResult: # Same as before
    auditor = QuerySafetyAuditor()
    is_dang, reason = auditor.audit(sql)
    return CustomScoreResult("Query Safety Audit", 1.0 if is_dang else 0.0, reason if is_dang else "Pass")
# --- END: Heuristic SQL Checks ---

def txt2sql_metrics(user_question: str, predicted_sql: str, db_schema: str) -> str:
    results = []
    print(f"\n>>> Evaluating Q: '{user_question[:50]}' SQL: {predicted_sql[:60].replace(chr(10),' ')}...")
    schema_map = parse_db_schema(db_schema)
    
    hallucination_result_obj = check_sql_column_hallucination(predicted_sql, schema_map)
    results.append(hallucination_result_obj.to_dict()) # metadata now includes extracted names for CSV
    
    results.append(check_query_safety(predicted_sql).to_dict())
    
    geval_intro = (f"Q:\"{user_question}\"\nSchema:\n{db_schema}") # Pass full schema to judge
    geval_crit = "Crit:Syntax,TblSel,ColSel(Semantic),Filter,Join,Agg,Completeness,Eff. Score 0-10."
    geval = CustomGEval(geval_intro, geval_crit, JUDGE_MODEL_ID)
    results.append(geval.score(output=predicted_sql).to_dict())
    return json.dumps(results, indent=2)

if __name__ == '__main__':
    print("--- Text-to-SQL Multi-Metric Evaluation Demo (Comprehensive) ---")
    # Schemas (ensure comprehensive_schema is well-defined for testing all normalizations)
    comprehensive_schema = """
    CREATE TABLE Employees ( EmployeeID INT PRIMARY KEY, FirstName VARCHAR(50), LastName VARCHAR(50), "E-mail" VARCHAR(100), job_id VARCHAR(10), SALARY DECIMAL(10,2));
    CREATE TABLE DEPARTMENTS ( DepartmentID INT PRIMARY KEY, "Department Name" VARCHAR(50) NOT NULL UNIQUE, ManagerID INT);
    CREATE TABLE "dbo"."Sales Orders" ("Order_ID" INT, Amount DECIMAL, Sales_Rep_ID INT); 
    CREATE VIEW EmployeeView AS SELECT EmployeeID, FirstName, LastName FROM Employees;
    """
    empty_schema = "-- No tables;"
    minimal_schema = "CREATE TABLE items (item_id INT, item_name TEXT);"
    cte_schema = "CREATE TABLE sales_data (region TEXT, amount INT, salesperson_id INT); CREATE TABLE sales_reps (rep_id INT, rep_name TEXT);"
    unparsable_schema = "CRATE TABEL oops (id INT);"

    test_cases_data = [
        # Basic Valid
        {"id":"tc001","uc":"Valid Select","sch":"comprehensive","q":"Emp names","sql":"SELECT FirstName, LastName FROM Employees;"},
        {"id":"tc002","uc":"Valid Quoted","sch":"comprehensive","q":"Dept names","sql":'SELECT "Department Name" FROM DEPARTMENTS;'},
        {"id":"tc003","uc":"Valid Schema.Table","sch":"comprehensive","q":"Order Amount","sql":'SELECT Amount FROM "dbo"."Sales Orders";'},
        {"id":"tc004","uc":"Valid Alias (source col valid)","sch":"comprehensive","q":"Emp Salary Alias","sql":"SELECT SALARY AS Remuneration FROM Employees;"},
        {"id":"tc005","uc":"Valid View","sch":"comprehensive","q":"Emp View","sql":"SELECT EmployeeID, FirstName FROM EmployeeView;"},
        
        # Column Hallucination
        {"id":"hc001","uc":"H: Unk Col","sch":"comprehensive","q":"Emp Age","sql":"SELECT FirstName, Age FROM Employees;"},
        {"id":"hc002","uc":"H: Unk Col Quoted","sch":"comprehensive","q":"Dept Building","sql":'SELECT "Department Name", "Building_No" FROM DEPARTMENTS;'},
        {"id":"hc003","uc":"H: Unk Table","sch":"comprehensive","q":"Products","sql":"SELECT ProductName FROM Products;"},
        {"id":"hc004","uc":"H: Query Empty Schema","sch":"empty","q":"Anything","sql":"SELECT col1 FROM anytable;"},
        {"id":"hc005","uc":"H: Query Unparsable Schema","sch":"unparsable","q":"Anything","sql":"SELECT col1 FROM oops;"}, # schema map empty
        {"id":"hc006","uc":"H: Col from View not in ViewDef (harder, GEval should catch)","sch":"comprehensive","q":"View Bad Col","sql":"SELECT EmployeeID, NonExistentColInView FROM EmployeeView;"}, #This heuristic check might pass if NonExistentColInView is not in parser.columns, GEval is better.

        # Query Safety
        {"id":"qs001","uc":"Safe: DROP","sch":"comprehensive","q":"N/A","sql":"DROP TABLE Employees;"},
        {"id":"qs002","uc":"Safe: UPDATE","sch":"comprehensive","q":"N/A","sql":"UPDATE Employees SET SALARY = 0;"},
        {"id":"qs003","uc":"Safe: INSERT","sch":"comprehensive","q":"N/A","sql":"INSERT INTO Employees (EmployeeID) VALUES (1);"},
        {"id":"qs004","uc":"Safe: EXEC","sch":"comprehensive","q":"N/A","sql":"SELECT exec('foo');"},

        # CTEs
        {"id":"cte001","uc":"CTE Valid","sch":"cte_schema","q":"CTE Sales","sql":"WITH RegionSales AS (SELECT region, SUM(amount) AS total_sales FROM sales_data GROUP BY region) SELECT region, total_sales FROM RegionSales;"},
        {"id":"cte002","uc":"CTE Unk Base Col","sch":"cte_schema","q":"CTE Bad Base","sql":"WITH BadSales AS (SELECT region, SUM(non_existent_col) FROM sales_data GROUP BY region) SELECT * FROM BadSales;"}, # non_existent_col from sales_data
        {"id":"cte003","uc":"CTE Unk Col from CTE","sch":"cte_schema","q":"CTE Bad CTE Col","sql":"WITH RegionSales AS (SELECT region, SUM(amount) AS total_sales FROM sales_data GROUP BY region) SELECT region, total_sales, imaginary_bonus FROM RegionSales;"}, # imaginary_bonus from RegionSales

        # Edge SQL
        {"id":"ec001","uc":"Simple Select 1","sch":"empty","q":"One","sql":"SELECT 1;"},
        {"id":"ec002","uc":"Unparsable SQL","sch":"comprehensive","q":"N/A","sql":"SELEC * FRM Employees"},
    ]

    df_test_cases = pd.DataFrame(test_cases_data)
    results_data_list = []
    schemas_dict = {"comprehensive": comprehensive_schema, "empty": empty_schema, "minimal": minimal_schema, "unparsable": unparsable_schema, "cte_schema": cte_schema}

    for idx, row in df_test_cases.iterrows():
        current_schema_str = schemas_dict.get(row['sch'], comprehensive_schema)
        metrics_json_str = txt2sql_metrics(row['q'], row['sql'], current_schema_str)
        metrics_list = json.loads(metrics_json_str)
        
        flat_row = {"test_id": row['id'], "use_case": row['uc'], "question": row['q'], 
                      "predicted_sql": row['sql'], "schema_name_used": row['sch']}
        
        # Add specific extracted data for CSV from hallucination metadata
        halluc_meta = next((m['metadata'] for m in metrics_list if m['name'] == "SQL Column Hallucination"), {})
        flat_row['ddl_extracted_tables'] = ", ".join(halluc_meta.get('parsed_ddl_tables', []))
        flat_row['ddl_extracted_table_columns_map_json'] = halluc_meta.get('parsed_ddl_columns_map_json', '{}')
        flat_row['sql_extracted_base_tables'] = ", ".join(halluc_meta.get('sql_extracted_base_tables', []))
        flat_row['sql_extracted_columns_for_ddl_validation'] = ", ".join(halluc_meta.get('sql_extracted_columns_for_ddl_validation', []))
        
        for metric in metrics_list: # Populate scores and reasons
            slug = metric['name'].lower().replace('(','').replace(')','').replace(' ', '_').replace('-', '_')
            flat_row[f"{slug}_score"] = metric.get('score'); flat_row[f"{slug}_reason"] = metric.get('reason')
        
        flat_row["full_metrics_json_output"] = metrics_json_str # Store the full JSON output as well
        results_data_list.append(flat_row)

    df_results = pd.DataFrame(results_data_list)
    cols_order = [
        "test_id","use_case","question","predicted_sql","schema_name_used", 
        "ddl_extracted_tables", "ddl_extracted_table_columns_map_json",
        "sql_extracted_base_tables", "sql_extracted_columns_for_ddl_validation",
        "sql_column_hallucination_score","sql_column_hallucination_reason", 
        "query_safety_audit_score","query_safety_audit_reason", 
        "llm_based_sql_evaluation_geval_score","llm_based_sql_evaluation_geval_reason",
        "full_metrics_json_output"
    ]
    for col in cols_order: # Ensure all columns exist
        if col not in df_results.columns: df_results[col] = None
    df_results = df_results[cols_order]

    csv_filename = "text2sql_evaluation_results_final.csv"
    try:
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Evaluation complete. Results saved to '{csv_filename}' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV: {e} ---"); print("Dumping results (first 5 rows):\n", df_results.head().to_string())
