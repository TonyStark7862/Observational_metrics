You are an expert SQL generator. Given a database schema and a question, write a {db_type} query.

Schema:
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
