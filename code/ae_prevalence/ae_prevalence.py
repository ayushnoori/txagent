"""
Adverse Event Prevalence Analysis Script

This script performs comprehensive adverse event prevalence analyses across different
disease cohorts, comorbidities, and drug exposures. Configuration for specific analyses
is defined in configurations.py.
"""

import math
import os
import re
import textwrap
from pathlib import Path

import duckdb

# To fix "Tcl_AsyncDelete: async handler deleted by the wrong thread" error.
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rich.console import Console
from rich.panel import Panel
from rich.progress import track, Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.text import Text

from configurations import CONFIGURATIONS

plt.rcParams['font.sans-serif'] = 'Arial'

# --- Script Configuration ---

# Initialize pretty printing console
console = Console()

# --- Global Paths & Settings ---

ROOT_DIR = Path("//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/txagent/")
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"
GROUPS_DIR = ROOT_DIR / "code" / "groups"
WINDOWS = [30, 90, None]

# --- Helper Functions ---

def slugify(text: str) -> str:
    """Converts text to a URL-friendly slug."""
    return "".join(ch.lower() if ch.isalnum() else '_' for ch in text).strip("_")

def clean_label(text: str) -> str:
    """Cleans a configuration string for display by lowercasing and replacing underscores."""
    return text.lower().replace("_", " ")

def _read_clean_groups(path: Path) -> pd.DataFrame:
    """
    Reads and cleans a group definition CSV.
    It also filters out any rows where the 'exclude' column is set to True.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            console.print(f"  [yellow]WARNING:[/] Definition file is empty: {path.name}")
            return pd.DataFrame() # Return empty DataFrame if file has no rows
    except pd.errors.EmptyDataError:
        console.print(f"  [yellow]WARNING:[/] Could not parse columns from file (it may be empty): {path.name}")
        return pd.DataFrame() # Return empty DataFrame on parsing error
    df.columns = [c.strip().lower() for c in df.columns]

    # Standardize all columns to stripped strings for consistent processing
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # If an 'exclude' column exists, drop rows where its value is 'True' (case-insensitive)
    if "exclude" in df.columns:
        # blank cells, 'False', or other values are kept.
        df = df[df["exclude"].str.lower() != 'true']

    if "name" in df.columns:
        df["name_key"] = df["name"].str.casefold()
        
    return df


def build_where_clause(rules: list) -> str:
    """Builds SQL OR-clauses from rule tuples."""
    parts = [f"({tc} = '{tv}' AND {c} LIKE '{pat}')" for c, tc, tv, pat in rules]
    return "(" + " OR ".join(parts) + ")"

def or_stats(a, b, c, d, add_halves=True):
    """Calculates odds ratio and 95% CI with Haldane-Anscombe correction."""
    if add_halves and (0 in (a, b, c, d)):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    if b == 0 or c == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    OR = (a * d) / (b * c)
    logOR = math.log(OR)
    SE = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lcl = math.exp(logOR - 1.96 * SE)
    ucl = math.exp(logOR + 1.96 * SE)
    
    return OR, logOR, SE, lcl, ucl

def ae_window_predicate(ae_col, idx_col, window_days):
    """Generates a SQL predicate for an AE occurring within a specified window."""
    if window_days is None:
        return f"{ae_col} > {idx_col}"
    return f"({ae_col} > {idx_col} AND {ae_col} <= {idx_col} + INTERVAL '{int(window_days)}' DAY)"

def smart_save_results(new_df: pd.DataFrame, file_path: Path, ae_column: str = "adverse_event") -> None:
    """
    Intelligently saves results by merging with existing data.
    
    - If file doesn't exist: saves new_df as-is
    - If file exists: reads existing data, replaces rows for AEs in new_df, 
      appends rows for new AEs not in existing data
    
    Args:
        new_df: DataFrame with new results to save
        file_path: Path where to save the CSV
        ae_column: Name of the column containing adverse event names (default: "adverse_event")
    """
    if new_df.empty:
        console.print(f"  [yellow]No data to save to {file_path.name}[/yellow]")
        return
    
    if not file_path.exists():
        # No existing file, just save
        new_df.to_csv(file_path, index=False)
        console.print(f"  [bold green]Saved[/bold green] {file_path.name} (new file, {len(new_df)} rows)")
        return
    
    # File exists, read and merge intelligently
    try:
        existing_df = pd.read_csv(file_path)
        
        if existing_df.empty:
            # Existing file is empty, just save new data
            new_df.to_csv(file_path, index=False)
            console.print(f"  [bold green]Saved[/bold green] {file_path.name} (replaced empty file, {len(new_df)} rows)")
            return
        
        # Get the adverse events in the new data
        new_aes = set(new_df[ae_column].unique())
        
        # Split existing data into two groups:
        # 1. Rows with AEs that will be replaced (exist in new_df)
        # 2. Rows with AEs that will be kept (don't exist in new_df)
        to_keep = existing_df[~existing_df[ae_column].isin(new_aes)]
        to_replace = existing_df[existing_df[ae_column].isin(new_aes)]
        
        # Combine: kept rows + new rows
        combined_df = pd.concat([to_keep, new_df], ignore_index=True)
        
        # Sort by adverse_event for consistency
        if ae_column in combined_df.columns:
            combined_df = combined_df.sort_values(ae_column).reset_index(drop=True)
        
        # Save combined data
        combined_df.to_csv(file_path, index=False)
        
        # Report what happened
        n_replaced = len(to_replace[ae_column].unique()) if not to_replace.empty else 0
        n_new = len(new_aes - set(to_replace[ae_column].unique()))
        n_kept = len(to_keep[ae_column].unique()) if not to_keep.empty else 0
        
        action_parts = []
        if n_replaced > 0:
            action_parts.append(f"replaced {n_replaced} AE(s)")
        if n_new > 0:
            action_parts.append(f"added {n_new} new AE(s)")
        if n_kept > 0:
            action_parts.append(f"kept {n_kept} existing AE(s)")
        
        action_desc = ", ".join(action_parts) if action_parts else "no changes"
        console.print(f"  [bold green]Saved[/bold green] {file_path.name} ({action_desc}, {len(combined_df)} total rows)")
        
    except Exception as e:
        console.print(f"  [bold red]ERROR:[/bold red] Could not read existing file {file_path.name}: {e}")
        console.print(f"  [yellow]Saving as backup to {file_path.stem}_backup.csv[/yellow]")
        backup_path = file_path.parent / f"{file_path.stem}_backup.csv"
        new_df.to_csv(backup_path, index=False)

# --- Plotting Functions ---

def format_cohort_label(label, width=15):
    """Formats cohort labels for plot axes."""
    segs = [s for s in re.split(r"\s*\+\s*", str(label).strip()) if s]
    lines = []
    n = len(segs)
    for i, seg in enumerate(segs):
        is_last = (i == n - 1)
        wrap_w = width if is_last else max(1, width - 2)
        wrapped = textwrap.wrap(seg, width=wrap_w, break_long_words=False)
        if not is_last and wrapped:
            wrapped[-1] += " +"
        lines.extend(wrapped)
    return "\n".join(lines)

def add_value_labels(ax, bars, numerators, denominators, percentages):
    """Places text labels with counts and percentages above each bar."""
    for rect, num, den, pct in zip(bars, numerators, denominators, percentages):
        if pd.notna(pct) and np.isfinite(pct):
            # Format label with comma separators for numbers and a newline for spacing
            label = f"{int(num):,}\nof {int(den):,}\n({pct:.2f}%)"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10  # Using a smaller font to prevent text overlap
            )
            
# --- Core Analysis Functions ---

def calculate_unadjusted_odds_ratios(con: duckdb.DuckDBPyConnection, adverse_events: list) -> pd.DataFrame:
    """Calculates unadjusted odds ratios for specified AEs in exposed vs. unexposed groups."""
    console.print(Panel("Step 4a: Calculate Unadjusted Odds Ratios", title_align="left", border_style="blue"))
    
    with console.status("[b]Defining analysis sets...[/b]", spinner="dots"):
        con.execute("CREATE OR REPLACE TEMP TABLE analysis_set AS SELECT * FROM patient_flags WHERE base_disease = 1 AND comorbidity = 1;")
        con.execute("CREATE OR REPLACE TEMP TABLE exposed_riskset AS SELECT patient_id, GREATEST(base_date, comorb_date, drug_date) AS index_date FROM analysis_set WHERE drug = 1;")
        con.execute("CREATE OR REPLACE TEMP TABLE unexposed_riskset AS SELECT patient_id, GREATEST(base_date, comorb_date) AS index_date FROM analysis_set WHERE drug = 0;")
    
    total_exp = con.execute("SELECT COUNT(*) FROM exposed_riskset;").fetchone()[0]
    total_unexp = con.execute("SELECT COUNT(*) FROM unexposed_riskset;").fetchone()[0]
    console.print(f"  [green]Exposed starters (D+D2+C):[/green] {total_exp:,}")
    console.print(f"  [green]Unexposed starters (D+D2, no C):[/green] {total_unexp:,}")

    if total_exp == 0 or total_unexp == 0:
        console.print("  [bold yellow]WARNING:[/] Cannot calculate odds ratios with zero patients in an analysis group.")
        return pd.DataFrame()

    rows = []
    for ae in track(adverse_events, description="Computing unadjusted ORs for AEs..."):
        ae_name, ae_slug = ae["name"], slugify(ae["name"])
        ae_where = build_where_clause(ae["rules"])
        con.execute(f"CREATE OR REPLACE TEMP TABLE ae_{ae_slug} AS SELECT patient_id, MIN(diagnosis_date) AS ae_date FROM dx WHERE {ae_where} GROUP BY patient_id;")

        for W in WINDOWS:
            win_label = "any_after_index" if W is None else f"{W}d"
            pred_exp = ae_window_predicate("a.ae_date", "e.index_date", W)
            pred_unx = ae_window_predicate("a.ae_date", "u.index_date", W)

            a = con.execute(f"SELECT COUNT(DISTINCT e.patient_id) FROM exposed_riskset e JOIN ae_{ae_slug} a USING (patient_id) WHERE {pred_exp};").fetchone()[0]
            b = total_exp - a
            c = con.execute(f"SELECT COUNT(DISTINCT u.patient_id) FROM unexposed_riskset u JOIN ae_{ae_slug} a USING (patient_id) WHERE {pred_unx};").fetchone()[0]
            d = total_unexp - c
            
            OR, logOR, SE, lcl, ucl = or_stats(a, b, c, d)
            
            rows.append({
                "adverse_event": clean_label(ae_name), "window": win_label,
                "a_exposed_E": a, "b_exposed_noE": b, "c_unexposed_E": c, "d_unexposed_noE": d,
                "total_exposed": total_exp, "total_unexposed": total_unexp,
                "odds_ratio": OR, "log_or": logOR, "se_log_or": SE, "ci95_low": lcl, "ci95_high": ucl
            })
            
    return pd.DataFrame(rows).sort_values(["adverse_event", "window"]).reset_index(drop=True)

def calculate_adjusted_odds_ratios(con: duckdb.DuckDBPyConnection, adverse_events: list, confounder_definitions: list) -> (pd.DataFrame, list):
    """Calculates odds ratios adjusted for confounders using logistic regression."""
    console.print(Panel("Step 4b: Calculate Confounder-Adjusted Odds Ratios", title_align="left", border_style="blue"))
    
    with console.status("[b]Building base regression dataset...[/b]", spinner="dots"):
        con.execute("""
            CREATE OR REPLACE TEMP TABLE regression_base AS
            (SELECT patient_id, 1 AS exposed, GREATEST(base_date, comorb_date, drug_date) AS index_date FROM patient_flags WHERE base_disease = 1 AND comorbidity = 1 AND drug = 1)
            UNION ALL
            (SELECT patient_id, 0 AS exposed, GREATEST(base_date, comorb_date) AS index_date FROM patient_flags WHERE base_disease = 1 AND comorbidity = 1 AND drug = 0);
        """)

        base_query = """
            SELECT
                r.patient_id, r.exposed, r.index_date, p.sex, p.socioeconomic_status,
                DATE_DIFF('year', p.birth_date, r.index_date) AS age_at_index
            FROM regression_base r
            JOIN pop p ON r.patient_id = p.patient_id
        """
        con.execute(f"CREATE OR REPLACE TEMP TABLE regression_data AS ({base_query});")

    # Add specified confounders as binary flags
    confounder_slugs = []
    if confounder_definitions:
        for conf in track(confounder_definitions, description="Adding confounder flags..."):
            conf_name, conf_slug = conf["name"], slugify(conf["name"])
            conf_where = build_where_clause(conf["rules"])
            confounder_slugs.append(conf_slug)

            con.execute(f"CREATE OR REPLACE TEMP TABLE {conf_slug}_dates AS SELECT patient_id, MIN(diagnosis_date) as conf_date FROM dx WHERE {conf_where} GROUP BY patient_id;")
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE temp_regression_data AS
                SELECT
                    rd.*,
                    CASE WHEN c.conf_date IS NOT NULL AND c.conf_date < rd.index_date THEN 1 ELSE 0 END AS {conf_slug}
                FROM regression_data rd
                LEFT JOIN {conf_slug}_dates c ON rd.patient_id = c.patient_id;
            """)
            con.execute("DROP TABLE regression_data;")
            con.execute("ALTER TABLE temp_regression_data RENAME TO regression_data;")
            con.execute(f"DROP TABLE {conf_slug}_dates;")

    model_df = con.execute("SELECT * FROM regression_data;").fetchdf()

    if model_df.empty:
        console.print("  [bold yellow]WARNING:[/] Base data for regression is empty. Skipping analysis.")
        return pd.DataFrame(), []

    # Build the formula string before the loop
    confounder_slugs = [slugify(c['name']) for c in confounder_definitions]
    base_formula = "outcome ~ exposed + age_at_index + C(sex) + C(socioeconomic_status)"
    confounders_str = " + ".join(confounder_slugs)
    formula = f"{base_formula} + {confounders_str}" if confounders_str else base_formula
    
    console.print(Panel(f"[cyan]Using formula for all models:[/cyan]\n{formula}", title="Regression Formula", border_style="yellow"))

    all_results = []
    model_summaries = []

    # Progress bar layout
    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TextColumn("•"),
        TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
    ]
    
    with Progress(*progress_columns, console=console) as progress:
        # Create a single task for the outer loop (adverse events)
        ae_task = progress.add_task("[cyan]Running regressions...", total=len(adverse_events))

        for ae in adverse_events:
            ae_name, ae_slug = ae["name"], slugify(ae["name"])
            ae_where = build_where_clause(ae["rules"])

            # Update progress bar
            progress.update(ae_task, description=f"[cyan]AE: [bold]{clean_label(ae_name)}[/bold]")
            
            ae_dates_df = con.execute(f"SELECT patient_id, MIN(diagnosis_date) AS ae_date FROM dx WHERE {ae_where} GROUP BY patient_id;").fetchdf()
            
            merged_df = pd.merge(model_df, ae_dates_df, on='patient_id', how='left')
            merged_df['index_date'] = pd.to_datetime(merged_df['index_date'])
            merged_df['ae_date'] = pd.to_datetime(merged_df['ae_date'])

            for W in WINDOWS:
                win_label = "any_after_index" if W is None else f"{W}d"
                temp_df = merged_df.copy()
                
                if W is None:
                    temp_df['outcome'] = ((temp_df['ae_date'] > temp_df['index_date'])).astype(int)
                else:
                    window_days = pd.to_timedelta(W, unit='d')
                    temp_df['outcome'] = ((temp_df['ae_date'] > temp_df['index_date']) & (temp_df['ae_date'] <= temp_df['index_date'] + window_days)).astype(int)

                if temp_df['outcome'].sum() < 5:
                    progress.log(f"[yellow]Skipping model for '{ae_name}' ({win_label}): Insufficient outcome events ({temp_df['outcome'].sum()}).[/yellow]")
                    continue

                # Build the formula string dynamically
                base_formula = "outcome ~ exposed + age_at_index + C(sex) + C(socioeconomic_status)"
                confounders_str = " + ".join(confounder_slugs)
                formula = f"{base_formula} + {confounders_str}" if confounders_str else base_formula
                
                try:
                    model_vars = ['outcome', 'exposed', 'age_at_index', 'sex', 'socioeconomic_status'] + confounder_slugs
                    temp_df.dropna(subset=model_vars, inplace=True)

                    model = smf.logit(formula, data=temp_df).fit(maxiter=100, disp=0)
                    
                    params = model.params
                    conf = model.conf_int()
                    pvalues = model.pvalues
                    
                    adj_or = np.exp(params.get('exposed', np.nan))
                    ci_low = np.exp(conf.loc['exposed', 0])
                    ci_high = np.exp(conf.loc['exposed', 1])
                    p_value = pvalues.get('exposed', np.nan)
                    
                    all_results.append({
                        "adverse_event": clean_label(ae_name),
                        "window": win_label,
                        "adjusted_odds_ratio": adj_or,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p_value": p_value,
                        "formula": formula
                    })
                    
                    summary_title = f"Model Summary: AE={clean_label(ae_name)}, Window={win_label}, Formula: {formula}\n"
                    model_summaries.append(summary_title + model.summary().as_csv())

                except Exception as e:
                    progress.log(f"[bold red]ERROR:[/] Could not fit regression for '{ae_name}' ({win_label}). Reason: {e}")
            
            # Manually advance the progress bar after all windows for an AE are done
            progress.advance(ae_task)

    return pd.DataFrame(all_results), model_summaries

def calculate_prevalence_stats(con: duckdb.DuckDBPyConnection, config: dict, adverse_events: list) -> pd.DataFrame:
    """Calculates AE prevalence across three sequentially defined cohorts."""
    console.print(Panel("Step 5: Calculate Prevalence Statistics", title_align="left", border_style="blue"))
    denom_c1 = con.execute("SELECT COUNT(*) FROM cohort1;").fetchone()[0]
    denom_c2 = con.execute("SELECT COUNT(*) FROM cohort2;").fetchone()[0]
    denom_c3 = con.execute("SELECT COUNT(*) FROM cohort3;").fetchone()[0]
    denom_c4 = con.execute("SELECT COUNT(*) FROM cohort4;").fetchone()[0]
    denom_total = con.execute("SELECT COUNT(*) FROM pop;").fetchone()[0]

    rows = []
    disease_label = clean_label(config['disease'])
    comorbidity_label = clean_label(config['comorbidity'])
    
    drug_display_name = config.get("drug_group_name", ', '.join([clean_label(d) for d in config['drugs']]))
    if config.get("drug_group_name"):
        drug_display_name = clean_label(drug_display_name)
    
    for ae in track(adverse_events, description="Computing prevalence for AEs..."):
        ae_name, ae_slug = ae["name"], slugify(ae["name"])
        ae_label = clean_label(ae_name)
        ae_where = build_where_clause(ae["rules"])
        con.execute(f"CREATE OR REPLACE TEMP TABLE ae_{ae_slug} AS SELECT patient_id, MIN(diagnosis_date) AS ae_date FROM dx WHERE {ae_where} GROUP BY patient_id;")

        n_total = con.execute(f"SELECT COUNT(*) FROM ae_{ae_slug};").fetchone()[0]
        # c1 = D
        n1 = con.execute(f"SELECT COUNT(*) FROM cohort1 c JOIN ae_{ae_slug} a USING (patient_id) WHERE a.ae_date > c.base_date;").fetchone()[0]
        # c2 = D + D2
        n2 = con.execute(f"SELECT COUNT(*) FROM cohort2 c JOIN ae_{ae_slug} a USING (patient_id) WHERE a.ae_date > GREATEST(c.base_date, c.comorb_date);").fetchone()[0]
        # c3 = D + C
        n3 = con.execute(f"SELECT COUNT(*) FROM cohort3 c JOIN ae_{ae_slug} a USING (patient_id) WHERE a.ae_date > GREATEST(c.base_date, c.drug_date);").fetchone()[0]
        # c4 = D + D2 + C
        n4 = con.execute(f"SELECT COUNT(*) FROM cohort4 c JOIN ae_{ae_slug} a USING (patient_id) WHERE a.ae_date > GREATEST(c.base_date, c.comorb_date, c.drug_date);").fetchone()[0]

        rows.extend([
            {"adverse_event": ae_label, "cohort": "total", "n_with_AE": n_total, "denominator": denom_total, "prevalence_pct": (n_total / denom_total * 100.0) if denom_total else 0.0},
            {"adverse_event": ae_label, "cohort": f"{disease_label}", "n_with_AE": n1, "denominator": denom_c1, "prevalence_pct": (n1 / denom_c1 * 100.0) if denom_c1 else 0.0},
            {"adverse_event": ae_label, "cohort": f"{disease_label} + {comorbidity_label}", "n_with_AE": n2, "denominator": denom_c2, "prevalence_pct": (n2 / denom_c2 * 100.0) if denom_c2 else 0.0},
            {"adverse_event": ae_label, "cohort": f"{disease_label} + {drug_display_name}", "n_with_AE": n3, "denominator": denom_c3, "prevalence_pct": (n3 / denom_c3 * 100.0) if denom_c3 else 0.0},
            {"adverse_event": ae_label, "cohort": f"{disease_label} + {comorbidity_label} + {drug_display_name}", "n_with_AE": n4, "denominator": denom_c4, "prevalence_pct": (n4 / denom_c4 * 100.0) if denom_c4 else 0.0}
        ])

    return pd.DataFrame(rows)

def generate_plots(prevalence_results: pd.DataFrame, pop_ae_df: pd.DataFrame, config: dict, save_dir: Path):
    """Generates and saves bar plots for AE prevalence."""
    console.print(Panel("Step 6: Generate and Save Plots", title_align="left", border_style="blue"))
    disease_label = clean_label(config['disease'])
    comorbidity_label = clean_label(config['comorbidity'])
    drug_display_name = config.get("drug_group_name", ', '.join([clean_label(d) for d in config['drugs']]))
    if config.get("drug_group_name"):
        drug_display_name = clean_label(drug_display_name)
    
    # --- Standardize cohort names for consistent plotting ---
    # Define the dynamic names as they are created in the prevalence calculation step
    dynamic_c1 = disease_label
    dynamic_c2 = f"{disease_label} + {comorbidity_label}"
    dynamic_c3 = f"{disease_label} + {drug_display_name}"
    dynamic_c4 = f"{disease_label} + {comorbidity_label} + {drug_display_name}"

    # Map the dynamic names to the desired standard, consistent names
    cohort_map = {
        "population": "general population",
        dynamic_c1: "disease",
        dynamic_c2: "disease + comorbidity",
        dynamic_c3: "disease + drug",
        dynamic_c4: "disease + comorbidity + drug",
    }
    
    # Define the plotting order using the new standard names
    COHORT_ORDER = ["general population", "total", "disease", "disease + drug", "disease + comorbidity", "disease + comorbidity + drug"]
    
    available_AEs_pop = set(pop_ae_df["adverse_event"].unique())
    requested_AEs = set(prevalence_results["adverse_event"].unique())
    valid_AEs = requested_AEs & available_AEs_pop

    missing_aes = requested_AEs - valid_AEs
    for ae in missing_aes:
        console.print(f"  [bold yellow]WARNING:[/] Population prevalence data not available for '{ae}'. It will be excluded from plots.")
    
    res_use = prevalence_results[prevalence_results["adverse_event"].isin(valid_AEs)]
    pop_use = pop_ae_df[pop_ae_df["adverse_event"].isin(valid_AEs)]
    
    results_pop = pd.concat([res_use, pop_use], ignore_index=True)

    # Create the new standardized cohort column using the map
    results_pop["cohort_standard"] = results_pop["cohort"].map(cohort_map)
    
    # Fill in non-mapped values (e.g., "total") from the original column
    results_pop["cohort_standard"] = results_pop["cohort_standard"].fillna(results_pop["cohort"])
    
    # Apply the categorical ordering to the new column
    results_pop["cohort_standard"] = pd.Categorical(results_pop["cohort_standard"], categories=COHORT_ORDER, ordered=True)
    results_pop = results_pop.sort_values(["adverse_event", "cohort_standard"]).reset_index(drop=True)
    results_pop.to_csv(save_dir / "prevalence_with_population.csv", index=False)
    
    plot_dir = save_dir / "barplots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Filter out the "total" cohort for plotting
    plot_data = results_pop[~(results_pop["cohort_standard"] == "total")].copy()
    
    bar_colors = ["#e5e5e5", "#e9c46a", "#f4a261", "#f4a261", "#e76f51"]

    for ae, g in track(plot_data.groupby("adverse_event", sort=True), description="Generating plots..."):
        g = g.sort_values("cohort_standard")
        fig, ax = plt.subplots(figsize=(5, 4)) # Slightly wider and taller for better spacing
        x = np.arange(len(g))
        
        # Prepare data for plotting and labeling
        numerators = g["n_with_AE"].values
        denominators = g["denominator"].values
        percentages = g["prevalence_pct"].astype(float).values
        
        bars = ax.bar(x, percentages, zorder=3, color=bar_colors, edgecolor="black")
        
        # Call the new add_value_labels function with all necessary data
        add_value_labels(ax, bars, numerators, denominators, percentages)
        ax.set_xticks(x)
        ax.set_xticklabels([format_cohort_label(c) for c in g["cohort_standard"].astype(str)], ha="center")
        ax.set_ylabel("Prevalence (%)", fontsize=12)
        ax.set_xlabel("Cohort", fontsize=12)
        
        # title_text = f"$\\bf{{Adverse\\ event:}}$ {str(ae).lower()}\n$\\bf{{Disease:}}$ {disease_label}\n$\\bf{{Comorbidity:}}$ {comorbidity_label}\n$\\bf{{Drug:}}$ {drug_display_name}"
        title_text = f"{str(ae).lower()}\nin {disease_label} + {comorbidity_label} + {drug_display_name}"
        ax.set_title(title_text, loc='left', pad=20, fontsize=12)
        
        ax.grid(axis='y', linestyle=":", linewidth=0.5, zorder=0)
        ax.margins(y=0.25) # Increase top margin to ensure labels fit
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        
        plot_base_name = slugify(f"{ae}-prevalence-plot")
        fig.savefig(plot_dir / f"{plot_base_name}.png", dpi=300)
        fig.savefig(plot_dir / f"{plot_base_name}.pdf")
        # plt.close(fig)
    
    console.print(f"  [bold green]Saved[/bold green] barplots.")

# --- Main Execution Block ---

def process_configuration(config: dict, con: duckdb.DuckDBPyConnection, group_dfs: dict, pop_ae_df: pd.DataFrame):
    """Runs the entire analysis pipeline for a single configuration."""
    title = f"Running Analysis: {clean_label(config['disease'])} + {clean_label(config['comorbidity'])}"
    content = Text(no_wrap=True)
    content.append("  Disease (D): ", style="green"); content.append(f"{clean_label(config['disease'])}\n")
    content.append("  Comorbidity (D2): ", style="green"); content.append(f"{clean_label(config['comorbidity'])}\n")

    if config.get("drug_group_name"):
        content.append("  Drug Group (C): ", style="green"); content.append(f"{config['drug_group_name']}\n")
        content.append("  Individual Drugs: ", style="green"); content.append(f"{', '.join(config['drugs'])}\n")
    else:
        drug_label = "Drug (C)" if len(config['drugs']) == 1 else "Drugs (C)"
        content.append(f"  {drug_label}: ", style="green"); content.append(f"{', '.join(config['drugs'])}\n")
        
    content.append("  Adverse Events (E): ", style="green"); content.append(f"{', '.join(config['aes'])}")
    console.print(Panel(content, title=title, border_style="cyan", title_align="left"))
    
    # === Step 1: Load and Validate Selections ===
    console.print(Panel("Step 1: Load and Validate Selections", title_align="left", border_style="blue"))
    disease, comorbidity, drugs, aes = config["disease"], config["comorbidity"], config["drugs"], config["aes"]
    sel_disease, sel_comorb = [disease.strip().casefold()], [comorbidity.strip().casefold()]
    sel_drugs, sel_aes = [d.strip().casefold() for d in drugs], [a.strip().casefold() for a in aes]
    sel_confounders = [c.strip().casefold() for c in config.get("confounders", [])]

    def _rules_from(df, selected_names):
        sub = df[df["name_key"].isin(selected_names)].drop_duplicates()
        return [(r.col, r.type_col, r.type_val, r.like_pattern) for r in sub.itertuples(index=False)]

    with console.status("[b]Loading definitions and rules...[/b]", spinner="dots"):
        base_disease_rules = _rules_from(group_dfs["base"], sel_disease)
        comorbidity_rules = _rules_from(group_dfs["comorbidity"], sel_comorb)
        drug_codes = group_dfs["drugs"][group_dfs["drugs"]["name_key"].isin(sel_drugs)]["atc5_code"].dropna().unique().tolist()
        
        adverse_events = []
        for ae_name, grp in group_dfs["ae"][group_dfs["ae"]["name_key"].isin(sel_aes)].groupby("name_key", sort=True):
            rules = [(r.col, r.type_col, r.type_val, r.like_pattern) for r in grp.itertuples(index=False)]
            adverse_events.append({"name": grp['name'].iloc[0], "rules": rules})

        confounder_definitions = []
        if sel_confounders and "confounders" in group_dfs:
            for conf_name, grp in group_dfs["confounders"][group_dfs["confounders"]["name_key"].isin(sel_confounders)].groupby("name_key", sort=True):
                rules = [(r.col, r.type_col, r.type_val, r.like_pattern) for r in grp.itertuples(index=False)]
                confounder_definitions.append({"name": grp['name'].iloc[0], "rules": rules})
    
    console.print(f"  [green]Disease rules found:[/green] {len(base_disease_rules)}")
    console.print(f"  [green]Comorbidity rules found:[/green] {len(comorbidity_rules)}")
    console.print(f"  [green]Drug ATC5 codes found:[/green] {len(drug_codes)}")
    console.print(f"  [green]Adverse events found:[/green] {len(adverse_events)}")
    console.print(f"  [green]Confounder definitions found:[/green] {len(confounder_definitions)}")
    
    for label, obj in [("DISEASE", base_disease_rules), ("COMORBIDITY", comorbidity_rules), ("DRUGS", drug_codes), ("AEs", adverse_events)]:
        if not obj:
            console.print(f"  [bold red]ERROR:[/] No entries found for '{label}'. Check names in config and CSVs. [bold]Skipping this run.[/bold]")
            return
    console.print("  [green]Selections validated successfully.[/green]")


    # === Step 2: Set Up Database Views ===
    console.print(Panel("Step 2: Set Up Database Views", title_align="left", border_style="blue"))
    cohort_dir = DATA_DIR / "cohorts" / disease
    pop_file = cohort_dir / f"population_{disease}.parquet"
    dx_file = cohort_dir / f"diagnoses_{disease}.parquet"
    med_file = cohort_dir / f"meds_{disease}.parquet"

    if not all([pop_file.exists(), dx_file.exists(), med_file.exists()]):
        console.print(f"  [bold red]ERROR:[/] Data files not found for disease '{disease}' in {cohort_dir}. [bold]Skipping.[/bold]")
        return
    
    # Define save directory
    drug_slug = slugify(config.get("drug_group_name") or '_'.join(drugs))
    save_dir_name = slugify(f"{disease}-{comorbidity}-{drug_slug}")
    save_dir = RESULTS_DIR / "ae_prevalence" / slugify(disease) / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with console.status("[b]Creating database views from Parquet files...[/b]", spinner="dots"):
        con.execute(f"CREATE OR REPLACE VIEW dx AS SELECT patient_id, time_stamp::DATE AS diagnosis_date, TRIM(code1) AS code1, code_type1 FROM read_parquet('{dx_file.as_posix()}');")
        con.execute(f"CREATE OR REPLACE VIEW meds AS SELECT patient_id, time_stamp::DATE AS rx_start_date, code1 AS atc5_code, type FROM read_parquet('{med_file.as_posix()}') WHERE type = 'Medications purchase';")
        latest_date = con.execute("SELECT MAX(diagnosis_date) FROM dx;").fetchone()[0]
        admin_end_date = latest_date.strftime('%Y-%m-%d')

        # Load population data into a Pandas DataFrame for imputation
        pop_df = pd.read_parquet(pop_file)
        # con.execute(f"CREATE OR REPLACE VIEW pop AS SELECT patient_id, date_of_birth::DATE AS birth_date, COALESCE(date_of_death::DATE, DATE '{admin_end_date}') AS observation_end_date FROM read_parquet('{pop_file.as_posix()}');")
        
        # --- Imputation of and categorization of socioeconomic_status ---
        console.print("  [yellow]Categorizing and imputing socioeconomic status...[/yellow]")
        n_missing_before = pop_df['socioeconomic_status'].isna().sum()

        # Create categorical SES levels based on population tertiles (~33% each).
        # NaNs in the original column will result in NaNs here.
        ses_cats = pd.qcut(
            pop_df['socioeconomic_status'],
            q=3,
            labels=['low', 'intermediate', 'high'],
            duplicates='drop'  # Handles non-unique bin edges
        )

        # Impute missing SES values by assigning them to the 'intermediate' category.
        # We use .astype('object') to allow filling with a string not in the original categories.
        pop_df['socioeconomic_status'] = ses_cats.astype('object').fillna('intermediate')

        # Ensure the column has a consistent categorical type
        ses_order = ['low', 'intermediate', 'high']
        pop_df['socioeconomic_status'] = pop_df['socioeconomic_status'].astype(pd.CategoricalDtype(categories=ses_order, ordered=True))
        
        n_imputed = n_missing_before
        console.print(f"  [green]Categorized SES into 3 levels and imputed {n_imputed:,} missing values to 'intermediate'.[/green]")

        # Define observation end date for the view
        pop_df['observation_end_date'] = pd.to_datetime(pop_df['date_of_death']).fillna(pd.to_datetime(admin_end_date))
        
        # Register the imputed Pandas DataFrame as a temporary DuckDB table
        con.register('pop_imputed', pop_df)
        
        # Create the final 'pop' view, now including sex and the imputed socioeconomic_status
        con.execute("""
            CREATE OR REPLACE VIEW pop AS 
            SELECT 
                patient_id, 
                date_of_birth::DATE AS birth_date,
                observation_end_date::DATE AS observation_end_date,
                sex,
                socioeconomic_status
            FROM pop_imputed;
        """)

    cohort_size = con.execute('SELECT COUNT(*) FROM pop;').fetchone()[0]
    console.print(f"  [green]Overall cohort size for '{disease}':[/green] {cohort_size:,}")

    # --- Demographic Summary ---
    console.print(Panel("Demographic Summary of Population", title_align="left", border_style="blue"))

    sex_counts = pop_df['sex'].value_counts()
    ses_counts = pop_df['socioeconomic_status'].value_counts()

    demographics_table = Table(title=f"Population Demographics for '{clean_label(disease)}'")
    demographics_table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
    demographics_table.add_column("Value", justify="left", style="magenta")

    demographics_table.add_row("Total Patients", f"{len(pop_df):,}")
    if 'F' in sex_counts:
        demographics_table.add_row("Females", f"{sex_counts['F']:,} ({sex_counts['F']/len(pop_df)*100:.1f}%)")
    if 'M' in sex_counts:
        demographics_table.add_row("Males", f"{sex_counts['M']:,} ({sex_counts['M']/len(pop_df)*100:.1f}%)")

    demographics_table.add_section()
    demographics_table.add_row("[bold]SES Category[/bold]", "")
    if 'low' in ses_counts:
        demographics_table.add_row("  Low", f"{ses_counts['low']:,} ({ses_counts['low']/len(pop_df)*100:.1f}%)")
    if 'intermediate' in ses_counts:
        demographics_table.add_row("  Intermediate", f"{ses_counts['intermediate']:,} ({ses_counts['intermediate']/len(pop_df)*100:.1f}%)")
    if 'high' in ses_counts:
        demographics_table.add_row("  High", f"{ses_counts['high']:,} ({ses_counts['high']/len(pop_df)*100:.1f}%)")
        
    console.print(demographics_table)

    # --- Save Demographics to CSV ---
    total_patients = len(pop_df)

    # Build the data as a list of rows
    demographics_rows = [
        {"Statistic": "Total Patients", "Value": total_patients, "Percentage": 100.0},
        {"Statistic": "Females", "Value": sex_counts.get('F', 0), "Percentage": (sex_counts.get('F', 0) / total_patients) * 100 if total_patients > 0 else 0},
        {"Statistic": "Males", "Value": sex_counts.get('M', 0), "Percentage": (sex_counts.get('M', 0) / total_patients) * 100 if total_patients > 0 else 0},
        {"Statistic": "Low SES", "Value": ses_counts.get('low', 0), "Percentage": (ses_counts.get('low', 0) / total_patients) * 100 if total_patients > 0 else 0},
        {"Statistic": "Intermediate SES", "Value": ses_counts.get('intermediate', 0), "Percentage": (ses_counts.get('intermediate', 0) / total_patients) * 100 if total_patients > 0 else 0},
        {"Statistic": "High SES", "Value": ses_counts.get('high', 0), "Percentage": (ses_counts.get('high', 0) / total_patients) * 100 if total_patients > 0 else 0},
    ]

    # Convert to DataFrame
    demographics_df = pd.DataFrame(demographics_rows)

    # Round percentages to 2 decimal places
    demographics_df["Percentage"] = demographics_df["Percentage"].round(2)

    # Save to CSV
    demographics_csv_path = save_dir / "demographic_summary.csv"
    demographics_df.to_csv(demographics_csv_path, index=False)

    # Console output
    console.print(
        f"  [bold green]Saved[/bold green] demographic summary to "
        f"[blue underline]{demographics_csv_path.relative_to(RESULTS_DIR)}[/blue underline]."
    )

    # === Step 3: Compute Index Dates and Define Cohorts ===
    console.print(Panel("Step 3: Compute Index Dates and Define Cohorts", title_align="left", border_style="blue"))
    with console.status("[b]Identifying first diagnosis/drug dates...[/b]", spinner="dots"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE base_disease AS SELECT patient_id, MIN(diagnosis_date) AS base_date FROM dx WHERE {build_where_clause(base_disease_rules)} GROUP BY patient_id;")
        con.execute(f"CREATE OR REPLACE TEMP TABLE comorbidity AS SELECT patient_id, MIN(diagnosis_date) AS comorb_date FROM dx WHERE {build_where_clause(comorbidity_rules)} GROUP BY patient_id;")
        con.execute(f"CREATE OR REPLACE TEMP TABLE drug_exposure AS SELECT patient_id, MIN(rx_start_date) AS drug_date FROM meds WHERE atc5_code IN {tuple(drug_codes)} GROUP BY patient_id;")

    with console.status("[b]Building master patient flag table...[/b]", spinner="dots"):
        con.execute("""
        CREATE OR REPLACE TEMP TABLE patient_flags AS
        SELECT p.patient_id,
            CASE WHEN bd.base_date IS NOT NULL THEN 1 ELSE 0 END AS base_disease, bd.base_date,
            CASE WHEN cd.comorb_date IS NOT NULL THEN 1 ELSE 0 END AS comorbidity, cd.comorb_date,
            CASE WHEN de.drug_date IS NOT NULL THEN 1 ELSE 0 END AS drug, de.drug_date
        FROM pop p
        LEFT JOIN base_disease bd USING (patient_id) LEFT JOIN comorbidity cd USING (patient_id) LEFT JOIN drug_exposure de USING (patient_id);
        """)
    
    with console.status("[b]Calculating cohort sizes...[/b]", spinner="dots"):
        n_d, n_d2, n_c, n_d_d2, n_d_c, n_d_d2_c = con.execute("""
            SELECT SUM(base_disease), 
                   SUM(comorbidity), 
                   SUM(drug),
                   SUM(CASE WHEN base_disease = 1 AND comorbidity = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN base_disease = 1 AND drug = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN base_disease = 1 AND comorbidity = 1 AND drug = 1 THEN 1 ELSE 0 END)
            FROM patient_flags;
        """).fetchone()
        con.execute("CREATE OR REPLACE TEMP TABLE cohort1 AS SELECT * FROM patient_flags WHERE base_disease = 1;")
        con.execute("CREATE OR REPLACE TEMP TABLE cohort2 AS SELECT * FROM patient_flags WHERE base_disease = 1 AND comorbidity = 1;")
        con.execute("CREATE OR REPLACE TEMP TABLE cohort3 AS SELECT * FROM patient_flags WHERE base_disease = 1 AND drug = 1;")
        con.execute("CREATE OR REPLACE TEMP TABLE cohort4 AS SELECT * FROM patient_flags WHERE base_disease = 1 AND comorbidity = 1 AND drug = 1;")

    table = Table(title="Cohort Sizes")
    table.add_column("Cohort Definition", justify="right", style="cyan", no_wrap=True)
    table.add_column("Patient Count", justify="right", style="magenta")
    table.add_row("N(D)", f"{n_d:,}")
    table.add_row("N(D2)", f"{n_d2:,}")
    table.add_row("N(C)", f"{n_c:,}")
    table.add_row("N(D + D2) [c2]", f"{n_d_d2:,}")
    table.add_row("N(D + C) [c3]", f"{n_d_c:,}")
    table.add_row("N(D + D2 + C) [c4]", f"{n_d_d2_c:,}")
    console.print(table)
    
    # === Steps 4-6: Run Analyses, Save Results, and Plot ===
    console.print(f"\n[bold]Results will be saved to:[/bold] [blue underline]{save_dir.relative_to(RESULTS_DIR)}[/blue underline]\n")

    unadjusted_or_results = calculate_unadjusted_odds_ratios(con, adverse_events)
    smart_save_results(unadjusted_or_results, save_dir / "unadjusted_OR_results.csv")
    
    if config.get("run_regression", True):
        adjusted_or_results, model_summaries = calculate_adjusted_odds_ratios(con, adverse_events, confounder_definitions)
        smart_save_results(adjusted_or_results, save_dir / "adjusted_OR_results.csv")
        if model_summaries:
            with open(save_dir / "regression_model_summaries.csv", "w") as f:
                f.write("\n\n".join(model_summaries))
            console.print("  [bold green]Saved[/bold green] regression model summaries.")

    prevalence_results = calculate_prevalence_stats(con, config, adverse_events)
    if not prevalence_results.empty:
        smart_save_results(prevalence_results, save_dir / "prevalence_results.csv")
        generate_plots(prevalence_results, pop_ae_df, config, save_dir)
    else:
        console.print("  [yellow]Skipping plotting due to empty prevalence results.[/yellow]")

def main():
    """Main function to orchestrate the analysis runs."""
    console.print(Panel("[bold magenta]Adverse Event Analysis Script[/bold magenta]", subtitle="Starting...", expand=False))

    with console.status("[b]Loading shared definition files...[/b]", spinner="dots"):
        group_files = {
            "base": GROUPS_DIR / "base.csv",
            "comorbidity": GROUPS_DIR / "comorbidities.csv",
            "drugs": GROUPS_DIR / "drugs.csv",
            "ae": GROUPS_DIR / "adverse_effects.csv",
            "confounders": GROUPS_DIR / "confounders.csv"
        }
        group_dfs = {}
        for name, path in group_files.items():
            if path.exists():
                group_dfs[name] = _read_clean_groups(path)
            else:
                if name == 'confounders':
                    console.print(f"[yellow]NOTE: '{path.name}' not found. Regression will only adjust for age, sex, and SES.[/yellow]")
                    group_dfs[name] = pd.DataFrame() # Create empty df to avoid key errors
                else:
                     console.print(f"[bold red]ERROR:[/] Essential definition file not found: {path}. Exiting.")
                     return
    console.print("Shared definition files loaded.")
    
    with console.status("[b]Loading population AE prevalence data...[/b]", spinner="dots"):
        pop_ae_file = ROOT_DIR / "data" / "codes" / "ae_patient_counts.csv"
        pop_ae_df = pd.read_csv(pop_ae_file)
        if pop_ae_df["prevalence_pct"].dropna().between(0, 1).all():
            pop_ae_df["prevalence_pct"] = pop_ae_df["prevalence_pct"] * 100.0
    console.print("Population AE prevalence data loaded.")

    con = duckdb.connect(database=":memory:")
    enabled_configs = [c for c in CONFIGURATIONS if c.get("enabled", True)]
    console.print(f"\n[bold]Found {len(enabled_configs)} enabled analysis configurations to run.[/bold]")
    
    for i, config in enumerate(CONFIGURATIONS):
        if not config.get("enabled", True):
            console.print(f"\n--- [yellow]Skipping Analysis {i+1}/{len(CONFIGURATIONS)} (disabled)[/yellow] ---")
            continue

        console.print(f"\n[bold magenta]━━━━━━━━━━ Analysis {i+1} of {len(CONFIGURATIONS)} ━━━━━━━━━━[/bold magenta]")
        try:
            process_configuration(config, con, group_dfs, pop_ae_df)
        except Exception as e:
            console.print(f"\n[bold red on white]ERROR during analysis for {config['disease']}: {e}[/]")
            import traceback
            console.print(traceback.format_exc())
            console.print("[bold red on white]Proceeding to next configuration.[/]")

    con.close()
    console.print("\n[bold magenta]--- Script finished. ---[/bold magenta]")

if __name__ == "__main__":
    main()
