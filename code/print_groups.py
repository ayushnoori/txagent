"""
Script to print all unique disease names (base.csv), comorbidities, and drugs
from the group definition files used in the AE prevalence analysis.
"""

import argparse
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize pretty printing console
console = Console()

# Define paths
ROOT_DIR = Path("//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/")
GROUPS_DIR = ROOT_DIR / "code" / "groups"

def read_clean_groups(path: Path) -> pd.DataFrame:
    """
    Reads and cleans a group definition CSV.
    Filters out any rows where the 'exclude' column is set to True.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            console.print(f"  [yellow]WARNING:[/] Definition file is empty: {path.name}")
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        console.print(f"  [yellow]WARNING:[/] Could not parse columns from file: {path.name}")
        return pd.DataFrame()
    except FileNotFoundError:
        console.print(f"  [red]ERROR:[/] File not found: {path}")
        return pd.DataFrame()
    
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standardize all columns to stripped strings
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    
    # If an 'exclude' column exists, drop rows where its value is 'True'
    if "exclude" in df.columns:
        df = df[df["exclude"].str.lower() != 'true']
    
    return df

def print_unique_names(df: pd.DataFrame, category: str):
    """Prints unique names from a DataFrame in a formatted table."""
    if df.empty or "name" not in df.columns:
        console.print(f"[yellow]No data available for {category}[/yellow]\n")
        return
    
    unique_names = sorted(df["name"].dropna().unique())
    
    table = Table(title=f"{category} ({len(unique_names)} unique entries)")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Name", justify="left", style="green")
    
    for idx, name in enumerate(unique_names, 1):
        table.add_row(str(idx), name)
    
    console.print(table)
    console.print()

def print_unique_names_csv(df: pd.DataFrame, category: str):
    """Prints unique names from a DataFrame as comma-separated list."""
    if df.empty or "name" not in df.columns:
        print(f"{category}: No data available")
        return
    
    unique_names = sorted(df["name"].dropna().unique())
    print(f"{category}: {', '.join(unique_names)}")

def main():
    """Main function to print all unique names from group definition files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Print all unique disease names, comorbidities, and drugs from group definition files."
    )
    parser.add_argument(
        "--csv", 
        action="store_true",
        help="Print as comma-separated lists instead of formatted tables"
    )
    args = parser.parse_args()
    
    # Choose the appropriate printing function
    print_func = print_unique_names_csv if args.csv else print_unique_names
    
    # Only print header if not in CSV mode
    if not args.csv:
        console.print(Panel("[bold magenta]Group Definitions Summary[/bold magenta]", 
                           expand=True))
        console.print()
    
    # Define the files to read
    group_files = {
        "Diseases": GROUPS_DIR / "base.csv",
        "Comorbidities": GROUPS_DIR / "comorbidities.csv",
        "Drugs": GROUPS_DIR / "drugs.csv",
        "Adverse Effects": GROUPS_DIR / "adverse_effects.csv",
        "Confounders": GROUPS_DIR / "confounders.csv"
    }
    
    # Read and print each category
    for category, path in group_files.items():
        if args.csv:
            # Simple loading for CSV mode
            df = read_clean_groups(path)
        else:
            # Fancy loading with spinner for table mode
            with console.status(f"[b]Loading {category}...[/b]", spinner="dots"):
                df = read_clean_groups(path)
        print_func(df, category)
    
    # Only print footer if not in CSV mode
    if not args.csv:
        console.print("[bold green]Done![/bold green]")

if __name__ == "__main__":
    main()

