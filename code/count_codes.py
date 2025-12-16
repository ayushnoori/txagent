"""
An interactive command-line tool to quickly filter and view diagnosis or
medication code counts from CSV files.

This script allows a user to select a dataset (diagnoses or medications) and
then apply case-insensitive filters on the description, code, or code type
fields to see how many unique codes match the criteria.

Run with:
1. `./setup.py`
2. `uv run python code/count_codes_interactive.py`
"""
import sys
from pathlib import Path
from typing import Union

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# --- Script Configuration ---

# Initialize pretty printing console
console = Console()

# --- Global Paths & Settings ---

# Assuming the script is run from a project root where a 'data' directory exists.
# Modify this if your directory structure is different.
ROOT_DIR = Path("//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/txagent/")
DATA_DIR = ROOT_DIR / "data" / "codes"
DIAG_COUNTS_PATH = DATA_DIR / "diag_counts.csv"
MED_COUNTS_PATH = DATA_DIR / "med_counts.csv"

# --- Core Functions ---

def load_data(diag_path: Path, med_path: Path) -> tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    """Loads diagnosis and medication data from specified CSV paths."""
    console.print(Panel("Step 1: Load Code Count Data", title_align="left", border_style="blue"))
    try:
        diag_df = pd.read_csv(diag_path)
        console.print(f"  [green]Successfully loaded[/green] {len(diag_df):,} rows from [bold cyan]{diag_path.name}[/bold cyan].")
    except FileNotFoundError:
        console.print(f"  [bold red]ERROR:[/] Diagnosis counts file not found at [bold cyan]{diag_path}[/bold cyan].")
        diag_df = None

    try:
        med_df = pd.read_csv(med_path)
        # Pre-process the 'type' column for medications for better readability
        if 'type' in med_df.columns:
            med_df['type'] = med_df['type'].str.replace('Medications ', '', regex=False).str.lower()
        console.print(f"  [green]Successfully loaded[/green] {len(med_df):,} rows from [bold cyan]{med_path.name}[/bold cyan].")
    except FileNotFoundError:
        console.print(f"  [bold red]ERROR:[/] Medication counts file not found at [bold cyan]{med_path}[/bold cyan].")
        med_df = None

    return diag_df, med_df


def apply_filters(source_df: pd.DataFrame, search_desc: str, search_code: str, search_type: str) -> pd.DataFrame:
    """Applies a series of filters to the provided DataFrame."""
    df = source_df.copy()

    # Apply filters sequentially
    if search_desc:
        df = df[df["description"].str.contains(search_desc, case=False, na=False)]
    if search_code:
        # Ensure code column is treated as a string to avoid errors with numeric codes
        df = df[df["code1"].astype(str).str.contains(search_code, case=False, na=False)]
    if search_type:
        # This filter is primarily for diagnoses
        df = df[df["code_type1"].str.contains(search_type, case=False, na=False)]

    return df


def display_results_table(filtered_df: pd.DataFrame, dataset_name: str):
    """Displays the filtered DataFrame in a formatted table in the console."""
    
    if filtered_df.empty:
        console.print("\n[bold yellow]No matching codes found for the specified filters.[/bold yellow]")
        return

    table = Table(
        title=f"Filtered Results from '{dataset_name}'",
        caption=f"Found {len(filtered_df)} matching codes.",
        show_header=True,
        header_style="bold magenta",
    )
    
    display_limit = 50
    df_to_show = filtered_df.head(display_limit).sort_values("count", ascending=False).reset_index(drop=True)

    if dataset_name == "Medications":
        table.add_column("Code", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Type", style="purple")
        table.add_column("Count", justify="right", style="bright_blue")
        
        for row in df_to_show.itertuples(index=False):
            table.add_row(
                str(getattr(row, 'code1', 'N/A')),
                str(getattr(row, 'description', 'N/A')),
                str(getattr(row, 'type', 'N/A')),
                f"{getattr(row, 'count', 0):,}",
            )
    else: # Default to Diagnoses format
        table.add_column("Code", style="cyan", no_wrap=True)
        table.add_column("Code Type", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Count", justify="right", style="bright_blue")

        for row in df_to_show.itertuples(index=False):
            table.add_row(
                str(getattr(row, 'code1', 'N/A')),
                str(getattr(row, 'code_type1', 'N/A')),
                str(getattr(row, 'description', 'N/A')),
                f"{getattr(row, 'count', 0):,}",
            )
        
    console.print(table)
    if len(filtered_df) > display_limit:
        console.print(f"[italic]... and {len(filtered_df) - display_limit:,} more rows not shown.[/italic]")


# --- Main Execution Block ---

def main():
    """Main function to orchestrate the interactive query tool."""
    console.print(Panel("[bold magenta]Interactive Code Search Tool[/bold magenta]", subtitle="Starting...", expand=False))

    diag_df, med_df = load_data(DIAG_COUNTS_PATH, MED_COUNTS_PATH)

    if diag_df is None and med_df is None:
        console.print("\n[bold red on white]FATAL:[/] No data files could be loaded. Exiting script.[/bold red on white]")
        sys.exit(1)

    while True:
        console.print(Panel("Step 2: Choose Dataset to Query", title_align="left", border_style="blue"))
        
        choices = []
        if diag_df is not None:
            choices.append("d")
        if med_df is not None:
            choices.append("m")
        choices.append("q")
        
        prompt_text = "Query (d)iagnoses or (m)edications? (q to quit)"
        choice = Prompt.ask(prompt_text, choices=choices, default="d" if "d" in choices else "m")

        if choice == "q":
            break
        
        active_df, dataset_name = (diag_df, "Diagnoses") if choice == "d" else (med_df, "Medications")

        console.print(f"\n[bold]Querying the '{dataset_name}' dataset. Enter filter criteria below.[/bold]")
        console.print("[italic gray50]Press Enter to skip a filter.[/italic gray50]")

        desc_label = "drug name" if choice == 'm' else "description"
        search_desc = Prompt.ask(f"  🔎 Search by [bold yellow]{desc_label}[/bold yellow]")
        search_code = Prompt.ask("  🔎 Search by [bold cyan]code[/bold cyan]")

        if choice == 'd':
            search_type = Prompt.ask("  🔎 Search by [bold green]code type[/bold green] (e.g., ICD10)")
        else:
            search_type = ""

        with console.status("[b]Applying filters and preparing results...[/b]", spinner="dots"):
            filtered_results = apply_filters(active_df, search_desc, search_code, search_type)
        
        display_results_table(filtered_results, dataset_name)
        
        console.print("\n" + "—"*50)
        console.print("[bold]New query.[/bold]\n")

    console.print("\n[bold magenta]--- Script finished. ---[/bold magenta]")


if __name__ == "__main__":
    main()
