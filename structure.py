python main.py \
    -q data/eval_cases.csv \
    --data_dir data \
    -f prompts/prompt.md \ # Add the prompt file argument
    -o results/local_csv_eval_results_rich_prompt.csv \
    -m "my_custom_sql_gen_v1" \
    -p 1



your_project_root/
│
├── main.py                 # Main script to run evaluations (CLI)
│
├── eval/
│   ├── __init__.py         # Makes 'eval' a package (can be empty)
│   ├── eval.py             # Original comparison logic (normalize_table, compare_df, subset_df)
│   └── local_eval.py       # Helper for in-memory DB execution & comparison
│
├── runners/
│   ├── __init__.py         # Makes 'runners' a package (can be empty)
│   └── abc_runner.py       # Runner orchestrating the custom generator & local eval
│
├── utils/
│   ├── __init__.py         # Makes 'utils' a package (can be empty)
│   ├── llm_abc.py          # Wrapper for your custom abc_sql_generator
│   ├── questions.py        # Loads eval definitions, prepares prompt inputs
│   ├── gen_prompt.py       # Generates the detailed prompt string
│   └── aliases.py          # Utility for generating table aliases
│
├── data/                   # Directory for all input data
│   ├── eval_cases_rich.csv # Example evaluation definition file (with optional cols)
│   ├── customers.csv       # Example data file
│   ├── orders.csv          # Example data file
│   └── products.csv        # Example data file
│   └── ...                 # Other data CSVs referenced in eval_cases*.csv
│
├── prompts/                # Directory for prompt templates
│   └── prompt.md           # Example prompt template file
│
└── results/                # Directory for output files (created automatically)
    └── (output CSV files will appear here after running main.py)
