# utils/gen_prompt.py
from typing import Dict, List, Optional
import json
# Assuming aliases utility is kept if needed for prompt generation
from .aliases import generate_aliases, get_table_names

# Note: This version assumes schema_string is provided.
# It does not directly depend on defog_data here, but expects
# the caller (questions.py) to provide the necessary schema info.

def generate_prompt(
    prompt_file: str,
    question: str,
    schema_string: str, # Expect schema as a DDL string
    db_type: str = "sqlite", # Default to sqlite for local eval
    instructions: str = "",
    k_shot_prompt: str = "", # For few-shot examples if provided in CSV
    glossary: str = "", # For glossary terms if provided in CSV
    prev_invalid_sql: str = "", # For error correction loops if needed
    prev_error_msg: str = "", # For error correction loops if needed
    # Add other potential fields from original if needed (e.g., specific few-shot examples)
    # question_0: str = "",
    # query_0: str = "",
    # question_1: str = "",
    # query_1: str = "",
    # cot_instructions: str = "", # Chain-of-thought related
    table_aliases_str: str = "", # Pre-generated aliases string
    # public_data: bool = True, # Less relevant now
    # columns_to_keep: int = 40, # Pruning related, handle elsewhere if needed
    # shuffle_metadata: bool = False, # Schema shuffling, handle elsewhere if needed
) -> str:
    """
    Generates a detailed prompt string using a template file and various components.

    Args:
        prompt_file: Path to the prompt template file (string format).
                     '.json' format is not supported in this simplified version.
        question: The natural language question.
        schema_string: The database schema formatted as a DDL string.
        db_type: The target SQL dialect (e.g., 'sqlite').
        instructions: Specific instructions for the query generation.
        k_shot_prompt: String containing few-shot examples.
        glossary: String containing glossary terms and definitions.
        prev_invalid_sql: Previously generated invalid SQL (for correction).
        prev_error_msg: Error message from previous invalid SQL (for correction).
        table_aliases_str: Pre-computed string of table aliases.

    Returns:
        The fully formatted prompt string.
    """
    if prompt_file.endswith(".json"):
         raise NotImplementedError("JSON prompt templates are not supported in this simplified version. Use a .md or .txt template.")

    try:
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at {prompt_file}")
        # Provide a basic fallback template
        prompt_template = """Schema:
{table_metadata_string}

{table_aliases}

{glossary}

{k_shot_prompt}

Instructions:
{instructions}

Previous Attempt (if any):
Invalid SQL: {prev_invalid_sql}
Error: {prev_error_msg}

Generate a {db_type} query for the following question:
{user_question}

SQL Query:
"""
        print(f"Warning: Using basic fallback prompt template.")
    except Exception as e:
        print(f"Error reading prompt file {prompt_file}: {e}")
        raise

    # Format the prompt using the provided components
    try:
        full_prompt = prompt_template.format(
            user_question=question,
            db_type=db_type,
            instructions=instructions if instructions else "Generate the SQL query.", # Provide default
            table_metadata_string=schema_string,
            k_shot_prompt=k_shot_prompt if k_shot_prompt else "",
            glossary=glossary if glossary else "",
            prev_invalid_sql=prev_invalid_sql if prev_invalid_sql else "N/A",
            prev_error_msg=prev_error_msg if prev_error_msg else "N/A",
            table_aliases=table_aliases_str if table_aliases_str else "-- No aliases provided.",
            # Add other placeholders here if your template uses them
            # e.g., question_0=question_0, query_0=query_0, etc.
        )
    except KeyError as e:
         print(f"Warning: Placeholder {{{e}}} not found in prompt template file '{prompt_file}'. It will be ignored.")
         # Attempt to format ignoring missing keys (less safe)
         from string import Formatter
         fmt = Formatter()
         mapping = {
            "user_question": question, "db_type": db_type, "instructions": instructions,
            "table_metadata_string": schema_string, "k_shot_prompt": k_shot_prompt,
            "glossary": glossary, "prev_invalid_sql": prev_invalid_sql,
            "prev_error_msg": prev_error_msg, "table_aliases": table_aliases_str,
            # Add defaults for any other keys used in the template
         }
         try:
             full_prompt = fmt.vformat(prompt_template, (), mapping)
         except Exception as format_err:
              print(f"Error formatting prompt even after handling KeyError: {format_err}")
              # Fallback to a very basic prompt
              full_prompt = f"Schema:\n{schema_string}\n\nQuestion:\n{question}\n\nSQL Query:"

    return full_prompt

# Keep the schema-to-text utility if needed by questions.py
# or if you want to generate schema string within generate_prompt
# (though passing it in is cleaner separation)
def to_prompt_schema(
    md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
) -> str:
    """
    Return a DDL statement for creating tables from a metadata dictionary
    `md` has the following structure:
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    This is just for converting the dictionary structure of one's metadata into a string
    for pasting into prompts, and not meant to be used to initialize a database.
    seed is used to shuffle the order of the tables when not None
    """
    import numpy as np # Keep numpy import local if only used here

    md_create = ""
    table_names = list(md.keys())
    if seed:
        np.random.seed(seed)
        np.random.shuffle(table_names)
    for table in table_names:
        md_create += f"CREATE TABLE {table} (\n"
        columns = md[table]
        if seed:
            np.random.seed(seed)
            np.random.shuffle(columns)
        for i, column in enumerate(columns):
            col_name = column["column_name"]
            # if column name has spaces, wrap it in double quotes
            if " " in col_name:
                col_name = f'"{col_name}"'
            dtype = column["data_type"]
            col_desc = column.get("column_description", "").replace("\n", " ")
            if col_desc:
                col_desc = f" --{col_desc}"
            if i < len(columns) - 1:
                md_create += f"  {col_name} {dtype},{col_desc}\n"
            else:
                # avoid the trailing comma for the last line
                md_create += f"  {col_name} {dtype}{col_desc}\n"
        md_create += ");\n"
    return md_create
