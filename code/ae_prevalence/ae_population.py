"""
Adverse Event Population Prevalence Calculator

This script computes the baseline prevalence of adverse events (AEs) across the 
entire outpatient population. These population-level prevalence statistics serve 
as a reference point for comparing disease-specific cohort prevalences in the 
main ae_prevalence.py analysis.

Key Features:
- Processes outpatient data in 250 parquet chunks for memory efficiency
- Counts unique patients experiencing each adverse event
- Calculates prevalence percentages for the entire population
- Supports incremental computation (can add new AEs without reprocessing existing ones)
- Performs disjoint patient ID checks across chunks to validate data integrity
- Saves results to ae_patient_counts.csv for use in downstream analyses

Input:
- Adverse event definitions from groups/adverse_effects.csv
- Outpatient diagnoses data split across 250 chunks

Output:
- ae_patient_counts.csv: Population-level prevalence for all defined AEs
- ae_chunk_patient_overlap.csv: Optional overlap report for data quality checks

Usage:
    python ae_population.py

The script will interactively prompt whether to recompute existing AEs or only 
process new adverse events not yet in the results file.
"""

import pandas as pd
import duckdb
import pyarrow.parquet as pq
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from itertools import islice

# --- Script Configuration ---

# Initialize pretty printing console
console = Console()

# --- Global Paths & Settings ---

# Project directories from the notebook
ROOT_DIR = Path("//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/txagent/")
COHORT_DIR = Path("//10.100.117.220/Projects$/R01-MainResearch/R01-Clalit_harvard_data/export_2024/Outpatients")
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"
GROUPS_DIR = ROOT_DIR / "code" / "groups"
AE_RULES_CSV = GROUPS_DIR / "adverse_effects.csv"

# Number of parquet chunks in the cohort directory
N_CHUNKS = 250

# Disjoint check settings
DO_DISJOINT_CHECK = True
OVERLAP_LOG_EXAMPLES = 10

# --- Helper Functions ---

def load_relation(file_path: Path) -> duckdb.DuckDBPyRelation:
    """Reads a Parquet file and returns a DuckDB relation for querying."""
    data_pq = pq.read_table(file_path)
    return duckdb.from_arrow(data_pq)

def build_where_clause(rules: list) -> str:
    """Builds SQL OR-clauses from rule tuples."""
    parts = [f"({tc} = '{tv}' AND {c} LIKE '{pat}')" for c, tc, tv, pat in rules]
    return "(" + " OR ".join(parts) + ")"

# --- Core Analysis Function ---

def compute_population_prevalence():
    """
    Computes the prevalence of all adverse events across the entire outpatient population
    by processing data in chunks.
    """
    # === Step 1: Load and Filter Adverse Event Definitions ===
    console.print(Panel("Step 1: Load and Filter Adverse Event Definitions", title_align="left", border_style="blue"))

    try:
        ae_df = pd.read_csv(AE_RULES_CSV)
        ae_df.columns = [c.strip().lower() for c in ae_df.columns]
    except FileNotFoundError:
        console.print(f"  [bold red]ERROR:[/] AE rules file not found at: {AE_RULES_CSV}. Aborting.")
        return

    required_cols = {"name", "col", "type_col", "type_val", "like_pattern"}
    if not required_cols.issubset(ae_df.columns):
        missing = required_cols - set(ae_df.columns)
        console.print(f"  [bold red]ERROR:[/] Missing required columns in AE rules file: {sorted(missing)}. Aborting.")
        return

    with console.status("[b]Processing AE rules...[/b]", spinner="dots"):
        for c in ae_df.columns:
            ae_df[c] = ae_df[c].astype(str).str.strip()
        if "exclude" in ae_df.columns:
            initial_rows = len(ae_df)
            ae_df = ae_df[ae_df["exclude"].str.lower() != 'true']
            console.print(f"  Filtered out {initial_rows - len(ae_df)} rows based on the 'exclude' column.")
        AE_RULES = {name: list(group.loc[:, ["col", "type_col", "type_val", "like_pattern"]].itertuples(index=False, name=None))
                    for name, group in ae_df.groupby("name", sort=False)}

    console.print(f"  [green]Loaded and processed definitions for {len(AE_RULES)} AEs.[/green]")

    # === Step 1.5: Check for Existing Results and Get User Input ===
    output_file_path = DATA_DIR / "codes" / "ae_patient_counts.csv"
    existing_results_df = None
    AE_RULES_TO_RUN = AE_RULES

    if output_file_path.exists():
        console.print(f"\n[bold yellow]Existing results file found at:[/bold yellow] [blue underline]{output_file_path}[/blue underline]")
        existing_results_df = pd.read_csv(output_file_path)
        existing_aes = set(existing_results_df['adverse_event'].astype(str).str.strip())
        current_aes = set(AE_RULES.keys())

        already_computed_aes = current_aes & existing_aes
        new_aes_to_compute = current_aes - existing_aes
        
        recompute_choice = ''
        if already_computed_aes:
            console.print(f"  - Population prevalence has already been computed for [bold cyan]{len(already_computed_aes)}[/bold cyan] AEs.")
        if new_aes_to_compute:
            console.print(f"  - Found [bold green]{len(new_aes_to_compute)}[/bold green] new AEs to compute.")
        if not new_aes_to_compute and already_computed_aes:
            console.print("  - [green]No new AEs found.[/green] All defined AEs have existing results.")
        
        if already_computed_aes:
            while recompute_choice not in ['y', 'n']:
                recompute_choice = console.input(f"\n[bold]Do you want to re-compute the {len(already_computed_aes)} existing AEs? (y/n): [/bold]").lower().strip()

        if recompute_choice == 'y':
            console.print("[cyan]User chose YES. All AEs will be re-computed and the file will be overwritten.[/cyan]")
            existing_results_df = None  # Clear existing results to signal an overwrite
            AE_RULES_TO_RUN = AE_RULES
        elif recompute_choice == 'n':
            console.print("[cyan]User chose NO. Only new AEs will be computed.[/cyan]")
            if not new_aes_to_compute:
                console.print("[bold green]No new AEs to compute. Exiting.[/bold green]")
                return
            AE_RULES_TO_RUN = {ae: rules for ae, rules in AE_RULES.items() if ae in new_aes_to_compute}
    
    if not AE_RULES_TO_RUN:
        console.print("[yellow]No AEs to process. Exiting.[/yellow]")
        return
        
    # === Step 2: Process Population Chunks ===
    console.print(Panel(f"Step 2: Process Population Data for {len(AE_RULES_TO_RUN)} AEs", title_align="left", border_style="blue"))

    all_patient_ids, overlap_records = set(), []
    ae_patient_sets = {ae_name: set() for ae_name in AE_RULES_TO_RUN.keys()}

    for i in track(range(N_CHUNKS), description="Processing chunks..."):
        chunk_pop_file = COHORT_DIR / f"Chunk_{i}" / f"population_{i}.parquet"
        chunk_diag_file = COHORT_DIR / f"Chunk_{i}" / f"diagnoses_{i}.parquet"

        if not (chunk_pop_file.exists() and chunk_diag_file.exists()):
            console.print(f"  [yellow]WARNING:[/] Missing files for Chunk {i}; skipping.")
            continue
        
        chunk_diags_rel = load_relation(chunk_diag_file)
        if existing_results_df is None: # Only process population if we need total patient count
            chunk_pop_df = load_relation(chunk_pop_file).df()
            cur_set = set(chunk_pop_df['patient_id'].dropna().astype(str))
            if DO_DISJOINT_CHECK and (overlap := cur_set & all_patient_ids):
                overlap_records.append({'chunk': i, 'n_overlap': len(overlap), 'example_patient_ids': list(islice(overlap, OVERLAP_LOG_EXAMPLES))})
                console.print(f"  [yellow]WARNING:[/] DISJOINT CHECK Chunk {i} has {len(overlap)} patient IDs already seen.")
            all_patient_ids.update(cur_set)

        for ae_name, rules in AE_RULES_TO_RUN.items():
            where_clause = build_where_clause(rules)
            ae_ids_df = chunk_diags_rel.filter(where_clause).project('patient_id').distinct().df()
            if not ae_ids_df.empty:
                ae_patient_sets[ae_name].update(ae_ids_df["patient_id"].astype(str).tolist())
    
    console.print("  [green]Finished processing all chunks.[/green]")

    # === Step 3: Calculate Prevalence and Save Results ===
    console.print(Panel("Step 3: Calculate and Save Prevalence Results", title_align="left", border_style="blue"))

    if existing_results_df is not None:
        total_patients = existing_results_df['denominator'].iloc[0]
        console.print(f"  Using total patient count from existing file: {total_patients:,}")
    else:
        total_patients = len(all_patient_ids)
        console.print(f"  Total unique patients in cohort: {total_patients:,}")

    if DO_DISJOINT_CHECK and overlap_records:
        save_path = DATA_DIR / "codes" / "ae_chunk_patient_overlap.csv"
        pd.DataFrame(overlap_records).to_csv(save_path, index=False)
        console.print(f"  [bold green]Saved[/bold green] overlap report to: [blue underline]{save_path}[/blue underline]")
    
    new_ae_counts = pd.DataFrame(
        [(ae, len(pids)) for ae, pids in ae_patient_sets.items()],
        columns=["adverse_event", "n_with_AE"]
    )
    new_ae_counts["cohort"] = "population"
    new_ae_counts["denominator"] = total_patients
    new_ae_counts["prevalence_pct"] = (new_ae_counts["n_with_AE"] / total_patients) * 100 if total_patients > 0 else 0.0

    if existing_results_df is not None:
        final_df_to_save = pd.concat([existing_results_df, new_ae_counts], ignore_index=True)
        console.print("\n[bold]Appending new results to existing file...[/bold]")
    else:
        final_df_to_save = new_ae_counts
    
    final_df_to_save = final_df_to_save.sort_values("n_with_AE", ascending=False).reset_index(drop=True)
    final_df_to_save.to_csv(output_file_path, index=False)
    console.print(f"  [bold green]Saved[/bold green] AE patient counts to: [blue underline]{output_file_path}[/blue underline]")
    console.print("\nFinal Prevalence Counts:")
    console.print(final_df_to_save)

# --- Main Execution Block ---

def main():
    """Main function to orchestrate the analysis."""
    console.print(Panel("[bold magenta]Adverse Event Population Prevalence Script[/bold magenta]", subtitle="Starting...", expand=False))
    try:
        compute_population_prevalence()
    except Exception as e:
        console.print(f"\n[bold red on white]An unexpected error occurred: {e}[/]")
        import traceback
        console.print(traceback.format_exc())
    console.print("\n[bold magenta]--- Script finished. ---[/bold magenta]")

if __name__ == "__main__":
    main()
