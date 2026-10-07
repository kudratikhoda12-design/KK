"""Builds notebooks/01_results_walkthrough.ipynb (run from anywhere):  python notebooks/build_notebook.py
Execute it afterwards with:  jupyter nbconvert --to notebook --execute --inplace notebooks/01_results_walkthrough.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# Explainable Demand Forecasting & Inventory Optimization - results walkthrough\n\n"
       "This notebook only **reads** the artefacts produced by `python run_project.py` (tables in `outputs/tables`, figures in `outputs/figures`) "
       "and re-computes a few numbers with the project's own functions, so every statement can be traced to code. "
       "Costs are **hypothetical**; observed sales may underestimate true demand during stockouts."),
    code("import sys, pathlib\n"
         "ROOT = pathlib.Path.cwd().resolve().parent\n"
         "sys.path.insert(0, str(ROOT))\n"
         "import pandas as pd, numpy as np\n"
         "from IPython.display import Image, display, Markdown\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)\n"
         "T, F = ROOT / 'outputs' / 'tables', ROOT / 'outputs' / 'figures'\n"
         "def show(name, width=900): display(Image(filename=str(F / name), width=width))"),
    md("## 1. Data and sample\nM5 Walmart data (30,490 series x 1,941 days). 18 item-store series were drawn with TRAIN information only (one store per state, three demand tiers, seeded draws)."),
    code("sel = pd.read_csv(ROOT / 'data' / 'processed' / 'selected_series.csv')\n"
         "sel[['series_id','tier','cat_id','train_mean','train_zero_share','train_adi','train_cv2','train_sb_class']].round(2)"),
    code("show('eda_01_daily_rolling.png'); show('eda_06_intermittency_map.png', 650)"),
    md("## 2. Forecast accuracy (test period, pooled over 18 series)"),
    code("t = pd.read_csv(T / 'forecast_summary_test.csv')\n"
         "tab = t[t.H.isin([7,14,28])].pivot(index='model', columns='H', values='WAPE').round(2).sort_values(28)\n"
         "tab.columns = [f'{c}-day WAPE %' for c in tab.columns]\ntab"),
    code("best = pd.read_csv(T / 'best_model_by_horizon.csv')\n"
         "best[['H','best_by_validation_WAPE','best_by_test_WAPE','test_WAPE','runner_up_test_WAPE','validation_choice_equals_test_best']]"),
    code("show('fc_02_model_comparison_wape.png')"),
    md("### Is the difference real? Diebold-Mariano (XGBoost vs each model, Holm-adjusted)"),
    code("dm = pd.read_csv(T / 'dm_tests_panel.csv')\n"
         "x = dm[(dm.ref=='XGBoost') & (dm.loss=='abs')].pivot(index='model_b', columns='H', values='p_holm').round(3)\n"
         "x.columns = [f'p_Holm H={c}' for c in x.columns]; x"),
    md("## 3. Explainability (SHAP)"),
    code("imp = pd.read_csv(T / 'shap_importance.csv')\n"
         "imp[imp.H==7].head(8)[['feature','group','mean_abs_shap','share_of_total','corr(feature value, shap)']].round(3)"),
    code("show('shap_summary_H7.png', 750); show('shap_individual_H7_1.png', 750)"),
    md("## 4. Uncertainty"),
    code("cov = pd.read_csv(T / 'uncertainty_coverage_test.csv')\n"
         "cov[(cov.chosen_method) & cov.model.isin(['XGBoost','SARIMA','MovingAverage']) & cov.H.isin([7,28])].pivot_table(index=['model','H'], columns='level', values='coverage').round(3)"),
    code("show('unc_01_prediction_interval_H7.png')"),
    md("## 5. Inventory results (MEDIUM penalty, 95% target)"),
    code("inv = pd.read_csv(T / 'final_inventory_results.csv')\n"
         "inv[(inv.Cost_Scenario=='MEDIUM') & (inv.Service_Level==0.95)][['Policy','Forecast_Model','Lead_Time_days','Stockouts','Fill_Rate','Cycle_Service_Level','Average_Inventory','Holding_Cost','Ordering_Cost','Stockout_Cost','Total_Cost']].round(3)"),
    code("show('inv_05_total_cost_components.png'); show('inv_06_sensitivity_heatmaps.png')"),
    md("### Paired bootstrap, forecast-driven policy B vs historical baseline A (negative = B cheaper)"),
    code("p = pd.read_csv(T / 'inventory_paired_bootstrap_B_vs_A.csv')\n"
         "p[p.scenario=='MEDIUM'][['L','SL','policy_B','rel_diff_pooled','boot_ci_low','boot_ci_high','series_B_cheaper']].round(3)"),
    md("## 6. Does the most accurate forecast give the lowest cost?"),
    code("s = pd.read_csv(T / 'accuracy_vs_cost_summary.csv')\n"
         "s[s.SL==0.95][['scenario','L','best_by_test_WAPE','best_by_total_cost','spearman_testWAPE_vs_cost_excl_Naive']].round(2)"),
    code("show('inv_07_accuracy_vs_cost.png')"),
    md("## 7. Try it yourself: simulate one series with the project's functions\n"
       "The cell below re-runs the daily simulator for one series under the historical policy A and the forecast-driven policy B (the validation-selected model) "
       "and prints the realised metrics - the same code that produced the tables."),
    code("from src.forecast_core import load_context, ForecastBook\n"
         "from src import inventory_experiments as ie, inventory as invm, evaluation as ev\n"
         "ctx = load_context(); book = ForecastBook(ctx).load(); ev.select_ma_alias(book)\n"
         "sig = pd.read_csv(T / 'sigma_table.csv')\n"
         "inputs = ie.prepare_inputs(ctx)\n"
         "s_idx, L, SL = 6, 7, 0.95\n"
         "polB = pd.read_csv(T / 'headline_policy_B_models.csv').set_index('lead_time').loc[L, 'model']\n"
         "rows = []\n"
         "for pol in [ie.POLICY_A, polB]:\n"
         "    rop, S = ie.policy_arrays(ctx, book, inputs, pol, L, SL, sig)\n"
         "    led = invm.simulate(inputs['demand'][s_idx], rop[s_idx], S[s_idx], L, ie.initial_inventory(inputs, L)[s_idx])\n"
         "    m = invm.add_costs(invm.ledger_metrics(led, L), inputs['cost_params'][s_idx])\n"
         "    rows.append({'series': ctx.ids[s_idx], 'policy': pol, 'fill_rate': m['fill_rate'], 'stockout_days': m['stockout_days'], 'avg_inventory': m['avg_inventory'], 'total_cost_MEDIUM': m['total_cost_MEDIUM']})\n"
         "pd.DataFrame(rows).round(3)"),
    md("*A single series can favour either policy* - the pooled result is an average over 18 series. The column `series_B_cheaper` in "
       "`inventory_paired_bootstrap_B_vs_A.csv` (section 5) counts, per configuration, in how many of the 18 series policy B had the lower total cost; "
       "the bootstrap interval there quantifies how much the pooled difference could vary across series."),
    md("## 8. Take-aways (computed from the tables above)"),
    code("g = lambda L: inv[(inv.Cost_Scenario=='MEDIUM')&(inv.Service_Level==0.95)&(inv.Lead_Time_days==L)].set_index('Policy')\n"
         "for L in (3,7,14):\n"
         "    a, b = g(L).loc['A_historical','Total_Cost'], g(L).loc['B_forecast_driven','Total_Cost']\n"
         "    print(f'L={L:2d}: total cost B vs A = {100*(b/a-1):+.1f}%  (cycle service level A {100*g(L).loc[\"A_historical\",\"Cycle_Service_Level\"]:.0f}% -> B {100*g(L).loc[\"B_forecast_driven\",\"Cycle_Service_Level\"]:.0f}%)')\n"
         "print('Cells where the best forecast is also the cheapest policy:', int(s.same_best_test_accuracy_and_cost.sum()), 'of', len(s))"),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "01_results_walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
