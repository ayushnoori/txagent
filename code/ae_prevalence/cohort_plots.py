"""
Cohort Plotting Functions

This module contains functions for generating diagnostic plots for cohort analyses,
including prevalence bar plots, ATC-5 code distributions, and follow-up time distributions.
"""

import re
import textwrap
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

# Helper functions (duplicated to avoid circular imports)
def slugify(text: str) -> str:
    """Converts text to a URL-friendly slug."""
    return "".join(ch.lower() if ch.isalnum() else '_' for ch in text).strip("_")

def clean_label(text: str) -> str:
    """Cleans a configuration string for display by lowercasing and replacing underscores."""
    return text.lower().replace("_", " ")

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

def build_where_clause(rules: list) -> str:
    """Builds SQL OR-clauses from rule tuples."""
    parts = [f"({tc} = '{tv}' AND {c} LIKE '{pat}')" for c, tc, tv, pat in rules]
    return "(" + " OR ".join(parts) + ")"

console = Console()

def generate_bar_plots(prevalence_results: pd.DataFrame, pop_ae_df: pd.DataFrame, config: dict, save_dir: Path):
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
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(g))
        
        # Prepare data for plotting and labeling
        numerators = g["n_with_AE"].values
        denominators = g["denominator"].values
        percentages = g["prevalence_pct"].astype(float).values
        
        bars = ax.bar(x, percentages, zorder=3, color=bar_colors, edgecolor="black")
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

def _get_cohort_labels(config: dict) -> dict:
    """Helper function to get cohort labels from config."""
    disease_label = clean_label(config['disease'])
    comorbidity_label = clean_label(config['comorbidity'])
    drug_display_name = config.get("drug_group_name", ', '.join([clean_label(d) for d in config['drugs']]))
    if config.get("drug_group_name"):
        drug_display_name = clean_label(drug_display_name)
    
    # Define cohort labels (excluding general population)
    return {
        "cohort1": disease_label,
        "cohort2": f"{disease_label} + {comorbidity_label}",
        "cohort3": f"{disease_label} + {drug_display_name}",
        "cohort4": f"{disease_label} + {comorbidity_label} + {drug_display_name}"
    }

def plot_drug_distribution(con: duckdb.DuckDBPyConnection, config: dict, adverse_events: list, save_dir: Path):
    """Generates plots showing distribution of ATC-5 codes per patient by cohort and adverse event status."""
    console.print(Panel("Step 7a: Plot Drug Distribution", title_align="left", border_style="blue"))
    
    cohort_labels = _get_cohort_labels(config)
    plot_dir = save_dir / "cohort_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ATC-5 code count per patient (memory-efficient: aggregate in DuckDB)
    console.print("  [cyan]Computing ATC-5 code counts per patient...[/cyan]")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE patient_atc5_counts AS
        SELECT 
            patient_id,
            COUNT(DISTINCT atc5_code) AS n_atc5_codes
        FROM meds
        GROUP BY patient_id;
    """)
    
    # Generate plots for each adverse event
    for ae in track(adverse_events, description="Generating drug distribution plots..."):
        ae_name, ae_slug = ae["name"], slugify(ae["name"])
        ae_label = clean_label(ae_name)
        ae_where = build_where_clause(ae["rules"])
        
        # Create temporary table for this AE (reuse if already exists from previous steps)
        con.execute(f"CREATE OR REPLACE TEMP TABLE ae_{ae_slug} AS SELECT patient_id, MIN(diagnosis_date) AS ae_date FROM dx WHERE {ae_where} GROUP BY patient_id;")
        
        # Collect data for all cohorts, stratified by AE status
        plot_data = []
        plot_labels = []
        
        for cohort_name, cohort_label in cohort_labels.items():
            # Patients WITH the AE
            query_with_ae = f"""
                SELECT 
                    COALESCE(ac.n_atc5_codes, 0) AS n_atc5_codes
                FROM {cohort_name} c
                INNER JOIN ae_{ae_slug} ae ON c.patient_id = ae.patient_id
                LEFT JOIN patient_atc5_counts ac ON c.patient_id = ac.patient_id
            """
            df_with_ae = con.execute(query_with_ae).fetchdf()
            
            # Patients WITHOUT the AE
            query_without_ae = f"""
                SELECT 
                    COALESCE(ac.n_atc5_codes, 0) AS n_atc5_codes
                FROM {cohort_name} c
                LEFT JOIN ae_{ae_slug} ae ON c.patient_id = ae.patient_id
                LEFT JOIN patient_atc5_counts ac ON c.patient_id = ac.patient_id
                WHERE ae.patient_id IS NULL
            """
            df_without_ae = con.execute(query_without_ae).fetchdf()
            
            if not df_with_ae.empty:
                plot_data.append(df_with_ae['n_atc5_codes'].values)
                plot_labels.append(f"{cohort_label}\n(with AE)")
            
            if not df_without_ae.empty:
                plot_data.append(df_without_ae['n_atc5_codes'].values)
                plot_labels.append(f"{cohort_label}\n(no AE)")
        
        # Skip if no data
        if not plot_data:
            console.print(f"  [yellow]Skipping plot for '{ae_label}': No data found.[/yellow]")
            continue
        
        # Create the plot with subplots for histograms
        n_groups = len(plot_data)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Determine common x-axis range for all histograms
        all_values = np.concatenate(plot_data) if plot_data else np.array([])
        if len(all_values) > 0:
            x_min = max(0, np.min(all_values) - 1)
            x_max = np.max(all_values) + 1
            bins = np.linspace(x_min, x_max, 50)
        else:
            bins = 50
        
        # Use same colors as barplots
        bar_colors = ["#e5e5e5", "#e9c46a", "#f4a261", "#f4a261", "#e76f51"]
        
        for idx, (values, label) in enumerate(zip(plot_data, plot_labels)):
            if idx < len(axes):
                ax = axes[idx]
                color = bar_colors[idx % len(bar_colors)]
                ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
                ax.set_title(label, fontsize=10, pad=5)
                ax.set_xlabel("Number of Distinct ATC-5 Codes", fontsize=9)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.grid(axis='y', linestyle=":", linewidth=0.5, alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        for idx in range(len(plot_data), len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title
        fig.suptitle(ae_label, fontsize=14, y=0.995)
        
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save plot
        plot_base_name = slugify(f"{ae_label}_drug_distribution")
        fig.savefig(plot_dir / f"{plot_base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(plot_dir / f"{plot_base_name}.pdf", bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
    console.print(f"  [bold green]Saved[/bold green] drug distribution plots to {plot_dir.name}/")

def plot_follow_up_time(con: duckdb.DuckDBPyConnection, config: dict, adverse_events: list, save_dir: Path):
    """Generates plots showing distribution of follow-up time by cohort and adverse event status."""
    console.print(Panel("Step 7b: Plot Follow-up Time Distribution", title_align="left", border_style="blue"))
    
    cohort_labels = _get_cohort_labels(config)
    plot_dir = save_dir / "cohort_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for each adverse event
    for ae in track(adverse_events, description="Generating follow-up time plots..."):
        ae_name, ae_slug = ae["name"], slugify(ae["name"])
        ae_label = clean_label(ae_name)
        ae_where = build_where_clause(ae["rules"])
        
        # Create temporary table for this AE (reuse if already exists from previous steps)
        con.execute(f"CREATE OR REPLACE TEMP TABLE ae_{ae_slug} AS SELECT patient_id, MIN(diagnosis_date) AS ae_date FROM dx WHERE {ae_where} GROUP BY patient_id;")
        
        # Collect data for all cohorts, stratified by AE status
        plot_data = []
        plot_labels = []
        
        for cohort_name, cohort_label in cohort_labels.items():
            # Calculate follow-up time: days from index_date to observation_end_date
            # Index date is the GREATEST of base_date, comorb_date, drug_date for each cohort
            if cohort_name == "cohort1":
                index_date_expr = "c.base_date"
            elif cohort_name == "cohort2":
                index_date_expr = "GREATEST(c.base_date, c.comorb_date)"
            elif cohort_name == "cohort3":
                index_date_expr = "GREATEST(c.base_date, c.drug_date)"
            else:  # cohort4
                index_date_expr = "GREATEST(c.base_date, c.comorb_date, c.drug_date)"
            
            # Patients WITH the AE
            query_with_ae = f"""
                SELECT 
                    DATE_DIFF('day', {index_date_expr}, p.observation_end_date) AS follow_up_days
                FROM {cohort_name} c
                INNER JOIN ae_{ae_slug} ae ON c.patient_id = ae.patient_id
                JOIN pop p ON c.patient_id = p.patient_id
                WHERE {index_date_expr} IS NOT NULL
                  AND p.observation_end_date IS NOT NULL
                  AND DATE_DIFF('day', {index_date_expr}, p.observation_end_date) >= 0
            """
            df_with_ae = con.execute(query_with_ae).fetchdf()
            
            # Patients WITHOUT the AE
            query_without_ae = f"""
                SELECT 
                    DATE_DIFF('day', {index_date_expr}, p.observation_end_date) AS follow_up_days
                FROM {cohort_name} c
                LEFT JOIN ae_{ae_slug} ae ON c.patient_id = ae.patient_id
                JOIN pop p ON c.patient_id = p.patient_id
                WHERE {index_date_expr} IS NOT NULL
                  AND p.observation_end_date IS NOT NULL
                  AND DATE_DIFF('day', {index_date_expr}, p.observation_end_date) >= 0
                  AND ae.patient_id IS NULL
            """
            df_without_ae = con.execute(query_without_ae).fetchdf()
            
            if not df_with_ae.empty:
                # Convert to years for better interpretability
                plot_data.append((df_with_ae['follow_up_days'].values / 365.25))
                plot_labels.append(f"{cohort_label}\n(with AE)")
            
            if not df_without_ae.empty:
                # Convert to years for better interpretability
                plot_data.append((df_without_ae['follow_up_days'].values / 365.25))
                plot_labels.append(f"{cohort_label}\n(no AE)")
        
        # Skip if no data
        if not plot_data:
            console.print(f"  [yellow]Skipping follow-up time plot for '{ae_label}': No data found.[/yellow]")
            continue
        
        # Create the plot with subplots for histograms
        n_groups = len(plot_data)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Cap values at 60 years and create bins with outlier bin
        OUTLIER_THRESHOLD = 60.0
        plot_data_capped = []
        for values in plot_data:
            capped = np.where(values > OUTLIER_THRESHOLD, OUTLIER_THRESHOLD, values)
            plot_data_capped.append(capped)
        
        # Create bins: 49 bins from 0 to 60, plus one bin for outliers at 60
        # This creates 50 bins total (49 regular + 1 outlier bin)
        bins = np.linspace(0, OUTLIER_THRESHOLD, 50)
        # Add a small bin edge above 60 to properly capture all values capped at 60
        bins = np.append(bins, OUTLIER_THRESHOLD + 0.01)
        
        # Use same colors as barplots
        bar_colors = ["#e5e5e5", "#e9c46a", "#f4a261", "#f4a261", "#e76f51"]
        
        for idx, (values, label) in enumerate(zip(plot_data_capped, plot_labels)):
            if idx < len(axes):
                ax = axes[idx]
                color = bar_colors[idx % len(bar_colors)]
                ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
                ax.set_title(label, fontsize=10, pad=5)
                ax.set_xlabel("Follow-up Time (years)", fontsize=9)
                ax.set_ylabel("Frequency", fontsize=9)
                # Set x-axis limit to show up to 60, with a note for outliers
                ax.set_xlim(left=0, right=OUTLIER_THRESHOLD + 1)
                ax.grid(axis='y', linestyle=":", linewidth=0.5, alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        for idx in range(len(plot_data_capped), len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title
        fig.suptitle(ae_label, fontsize=14, y=0.995)
        
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save plot
        plot_base_name = slugify(f"{ae_label}_follow_up_time")
        fig.savefig(plot_dir / f"{plot_base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(plot_dir / f"{plot_base_name}.pdf", bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
    console.print(f"  [bold green]Saved[/bold green] follow-up time distribution plots to {plot_dir.name}/")

