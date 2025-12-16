"""
Cohort Construction Pipeline

This script generates patient cohorts from Clalit export data in a fully 
stream-processing, low-RAM fashion. It leverages DuckDB's file globbing and 
direct COPY TO capabilities to avoid creating large intermediate files on disk.

Process:
1. Loads cohort definitions from a CSV.
2. Builds a mapping table (patient_id -> cohort) in a local DuckDB instance.
3. Stream-exports filtered data (meds, diagnoses, etc.) for each cohort into
   separate Parquet files.

Run with:
    `uv run python code/build_diagnosis_cohorts.py`
"""
import sys
import duckdb
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# --- Script Configuration ---

# Initialize pretty printing console
console = Console()

# --- Global Paths & Settings ---

# Root paths
ROOT_DIR = Path("//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/txagent")
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "cohorts"

# Source directory containing the chunked Parquet files
COHORT_DIR = Path("//10.100.117.220/Projects$/R01-MainResearch/R01-Clalit_harvard_data/export_2024/Outpatients")

# Location for the DuckDB database file
DUCKDB_FILE = OUTPUT_DIR / "cohorts.duckdb"

# Configuration file for cohort rules
COHORTS_CSV = ROOT_DIR / "code" / "groups" / "base.csv"

# DuckDB Memory Limit
MEMORY_LIMIT = "2GB"

# Define the logical tables to export for each cohort
TABLES_TO_EXPORT = {
    "population": "population",
    "diagnoses": "diagnoses",
    "meds": "meds",
    "events": "events",
}

# --- Core Functions ---

def load_cohort_definitions(csv_path: Path) -> dict:
    """
    Reads the cohort CSV and builds a dictionary of rules.
    Structure: { 'cohort_name': [ (col, type_col, type_val, like_pattern), ... ] }
    """
    console.print(Panel("Step 1: Load Cohort Definitions", title_align="left", border_style="blue"))
    
    if not csv_path.exists():
        console.print(f"[bold red]ERROR:[/] Cohort CSV not found at [cyan]{csv_path}[/cyan]")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        cohorts = {}

        for _, row in df.iterrows():
            name = row['name']
            # Tuple structure: col, type_col, type_val, like_pattern
            entry = (row['col'], row['type_col'], row['type_val'], row['like_pattern'])
            cohorts.setdefault(name, []).append(entry)
            
        console.print(f"  [green]Successfully loaded[/green] [bold]{len(cohorts)}[/bold] cohort definitions from CSV.")
        return cohorts
        
    except Exception as e:
        console.print(f"[bold red]ERROR:[/] Failed to read CSV: {e}")
        sys.exit(1)


def init_duckdb(db_path: Path, memory_limit: str):
    """Initializes DuckDB connection and sets memory limits."""
    # Ensure output dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"  [dim]Connecting to DuckDB at: {db_path.name}[/dim]")
    con = duckdb.connect(database=str(db_path), read_only=False)
    
    # Set memory limit as per config screenshot
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    
    # Log version (mimicking screenshot output)
    version = con.execute('SELECT version()').fetchone()[0]
    console.print(f"  [dim]DuckDB Version: {version}[/dim]")
    console.print(f"  [dim]Memory Limit set to: {memory_limit}[/dim]")
    
    return con


def build_patient_cohort_table(con, cohorts: dict, source_dir: Path):
    """
    Scans all diagnosis files and builds the 'patient_cohort' mapping table
    based on the provided rules.
    """
    console.print(Panel("Step 2: Build Patient-Cohort Mapping", title_align="left", border_style="blue"))

    # Create the mapping table
    con.execute("""
        CREATE OR REPLACE TABLE patient_cohort (
            patient_id BIGINT,
            cohort VARCHAR,
            PRIMARY KEY (patient_id, cohort)
        );
    """)

    # Use glob pattern to read all diagnosis files at once
    all_diagnoses_path = str(source_dir / "Chunk_*" / "diagnoses_*.parquet")
    
    console.print(f"  [dim]Scanning files at: .../Chunk_*/diagnoses_*.parquet[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Building cohorts...", total=len(cohorts))

        for cohort_name, rules in cohorts.items():
            or_clauses = []
            
            # Construct the dynamic SQL WHERE clause from rules
            for col, type_col, type_val, pattern in rules:
                clause = (
                    f"(upper({type_col}) = '{str(type_val).upper()}' "
                    f"AND upper({col}) LIKE '{str(pattern).upper()}')"
                )
                or_clauses.append(clause)
            
            where_clause = " OR ".join(or_clauses)

            insert_query = f"""
                INSERT OR IGNORE INTO patient_cohort
                SELECT DISTINCT patient_id, '{cohort_name}' AS cohort
                FROM read_parquet('{all_diagnoses_path}')
                WHERE {where_clause};
            """
            
            con.execute(insert_query)
            progress.advance(task)

    # Validate count
    count = con.execute("SELECT COUNT(*) FROM patient_cohort").fetchone()[0]
    console.print(f"  [green]Mapping table built successfully.[/green] Total entries: [bold cyan]{count:,}[/bold cyan]")


def export_cohort_data(con, cohorts: dict, source_dir: Path, output_root: Path):
    """
    Stream-exports data for each cohort directly to final Parquet files.
    """
    console.print(Panel("Step 3: Exporting Data for Each Cohort", title_align="left", border_style="blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} cohorts"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[magenta]Exporting cohorts...", total=len(cohorts))

        for cohort_name in cohorts.keys():
            cohort_out_dir = output_root / cohort_name
            cohort_out_dir.mkdir(parents=True, exist_ok=True)

            for logical_name, file_stem in TABLES_TO_EXPORT.items():
                
                # Glob pattern for source files (e.g., meds_*.parquet)
                source_files = str(source_dir / "Chunk_*" / f"{file_stem}_*.parquet")
                output_file = str(cohort_out_dir / f"{logical_name}_{cohort_name}.parquet")

                # The COPY command streams results directly from query to file
                # union_by_name=True allows for slight schema drift in chunks if necessary
                copy_query = f"""
                    COPY (
                        SELECT t.*
                        FROM read_parquet('{source_files}', union_by_name=true) AS t
                        INNER JOIN patient_cohort pc ON t.patient_id = pc.patient_id
                        WHERE pc.cohort = '{cohort_name}'
                    ) TO '{output_file}' (FORMAT PARQUET);
                """

                try:
                    con.execute(copy_query)
                except duckdb.IOException as e:
                    console.print(f"    [yellow]Warning:[/yellow] Could not read files for {logical_name}. {e}")
                except Exception as e:
                    console.print(f"    [red]Error processing {logical_name} for {cohort_name}: {e}[/red]")

            progress.advance(main_task)

    console.print(f"\n[green]All done! Results are in:[/green] [bold]{output_root}[/bold]")

# --- Main Execution Block ---

def main():
    console.print(Panel("[bold magenta]Cohort Construction Pipeline[/bold magenta]", subtitle="DuckDB Stream Processing", expand=False))

    # Load definitions
    cohorts = load_cohort_definitions(COHORTS_CSV)

    # Setup database
    con = init_duckdb(DUCKDB_FILE, MEMORY_LIMIT)

    try:
        # Build mapping
        build_patient_cohort_table(con, cohorts, COHORT_DIR)

        # Export data
        export_cohort_data(con, cohorts, COHORT_DIR, OUTPUT_DIR)
        
    finally:
        con.close()
        console.print("\n[bold magenta]--- Script finished. ---[/bold magenta]")

if __name__ == "__main__":
    main()
