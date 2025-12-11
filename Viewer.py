import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import *

import re
from collections import defaultdict, deque

import pandas as pd


# ================== RELATIVE PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
SCAN_SCRIPT = os.path.join(BASE_DIR, "Scan.py")
# ====================================================

PBIX_STRUCTURE_CSV = "pbix_structure.csv"
MODEL_USAGE_CSV = "model_usage.csv"
RELATIONSHIPS_CSV = "model_relationships.csv"
OVERVIEW_XLSX = "model_overview.xlsx"


def compute_transitive_used(df_usage: pd.DataFrame):
    """
    Samme logik som i Scan.py / debug:
    Seed = rækker hvor UsedInFiles != "".
    Gå gennem DAX-udtryk og markér alt opstrøms som brugt.
    """
    df_usage = df_usage.fillna("")

    n = len(df_usage)
    deps = [set() for _ in range(n)]

    table_name_map = {}
    global_name_map = {}

    for i, row in df_usage.iterrows():
        name = str(row.get("Name", "") or "").strip()
        table = str(row.get("Table", "") or "").strip()
        if not name:
            continue

        name_l = name.lower()
        table_l = table.lower()

        table_name_map.setdefault((table_l, name_l), set()).add(i)
        global_name_map.setdefault(name_l, set()).add(i)

    bracket = re.compile(r"\[([^\]]+)\]")

    for i, row in df_usage.iterrows():
        expr = str(row.get("Expression", "") or "")
        if not expr:
            continue
        expr_low = expr.lower()
        table_i = str(row.get("Table", "") or "").strip().lower()

        for m in bracket.finditer(expr_low):
            token = m.group(1).strip().lower()
            if not token:
                continue

            idxs = table_name_map.get((table_i, token), set())
            if not idxs:
                idxs = global_name_map.get(token, set())

            for j in idxs:
                if i != j:
                    deps[i].add(j)

    used = [False] * n
    q = deque()

    for i, row in df_usage.iterrows():
        uf = str(row.get("UsedInFiles", "") or "").strip()
        if uf != "":
            used[i] = True
            q.append(i)

    while q:
        i = q.popleft()
        for j in deps[i]:
            if not used[j]:
                used[j] = True
                q.append(j)

    return used


class ModelData:
    """
    Wrapper for CSV'erne fra Scan.py
    """

    def __init__(self):
        self.output_dir = None

        self.df_struct = None       # pbix_structure.csv
        self.df_usage = None        # model_usage.csv
        self.df_rel = None          # model_relationships.csv

        self.tables = {}            # table -> {"objects": [row_indices]}
        self.reports = defaultdict(list)  # report_name -> [row_indices]

    def load(self, output_dir: str):
        self.output_dir = output_dir

        struct_path = os.path.join(output_dir, PBIX_STRUCTURE_CSV)
        usage_path = os.path.join(output_dir, MODEL_USAGE_CSV)
        rel_path = os.path.join(output_dir, RELATIONSHIPS_CSV)

        if not os.path.exists(struct_path):
            raise FileNotFoundError(struct_path)
        if not os.path.exists(usage_path):
            raise FileNotFoundError(usage_path)
        if not os.path.exists(rel_path):
            raise FileNotFoundError(rel_path)

        self.df_struct = pd.read_csv(struct_path, sep=";")
        self.df_usage = pd.read_csv(usage_path, sep=";").fillna("")
        self.df_rel = pd.read_csv(rel_path, sep=";").fillna("")

        # compute transitive + UsedAny
        used_trans = compute_transitive_used(self.df_usage)
        self.df_usage["UsedTransitive"] = ["Yes" if x else "No" for x in used_trans]
        self.df_usage["UsedAny"] = self.df_usage.apply(
            lambda r: "Yes"
            if (str(r.get("Used", "")).strip().lower() == "yes"
                or str(r.get("UsedTransitive", "")).strip().lower() == "yes")
            else "No",
            axis=1,
        )

        # tables -> rows
        self.tables.clear()
        for idx, row in self.df_usage.iterrows():
            table = str(row.get("Table", "") or "")
            if not table:
                continue
            d = self.tables.setdefault(table, {"objects": []})
            d["objects"].append(idx)

        # reports -> rows
        self.reports.clear()
        for idx, row in self.df_usage.iterrows():
            used_files = str(row.get("UsedInFiles", "") or "")
            if not used_files:
                continue
            for part in used_files.split("|"):
                p = part.strip()
                if p:
                    self.reports[p].append(idx)


class ViewerGUI:

    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("PBIX Model Viewer")
        self.root.geometry("1400x850")

        self.data = ModelData()
        self.output_dir = None

        # til DAX dependencies
        self.dax_deps = {}

        # --------- TOPBAR ---------
        top = tb.Frame(root, padding=5)
        top.pack(side=tk.TOP, fill=tk.X)

        tb.Label(top, text="Master PBIX:").grid(row=0, column=0, sticky="w")
        self.master_var = tk.StringVar()
        tb.Entry(top, textvariable=self.master_var, width=60).grid(row=0, column=1, sticky="we", padx=5)
        tb.Button(top, text="Browse", bootstyle=INFO, command=self.pick_master).grid(row=0, column=2, padx=5)

        tb.Label(top, text="Reports folder:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.reports_var = tk.StringVar()
        tb.Entry(top, textvariable=self.reports_var, width=60).grid(row=1, column=1, sticky="we", padx=5, pady=(4, 0))
        tb.Button(top, text="Browse", bootstyle=INFO, command=self.pick_reports_folder).grid(
            row=1, column=2, padx=5, pady=(4, 0)
        )

        tb.Button(top, text="Run Scan", bootstyle=SUCCESS, command=self.run_scan).grid(
            row=0, column=3, padx=(15, 5)
        )
        tb.Button(top, text="Load Results", bootstyle=PRIMARY, command=self.load_results).grid(
            row=1, column=3, padx=(15, 5)
        )

        top.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        tb.Label(root, textvariable=self.status_var, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        # --------- NOTEBOOK ---------
        self.notebook = tb.Notebook(root, bootstyle="primary")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_tables = tb.Frame(self.notebook)
        self.tab_reports = tb.Frame(self.notebook)
        self.tab_rel = tb.Frame(self.notebook)
        self.tab_dax = tb.Frame(self.notebook)

        self.notebook.add(self.tab_tables, text="Tables")
        self.notebook.add(self.tab_reports, text="Reports")
        self.notebook.add(self.tab_rel, text="Relations")
        self.notebook.add(self.tab_dax, text="DAX")

        # build tabs
        self.build_tables_tab()
        self.build_reports_tab()
        self.build_relations_tab()
        self.build_dax_tab()

    # ---------- TOPBAR actions ----------

    def pick_master(self):
        path = filedialog.askopenfilename(
            title="Select master PBIX",
            filetypes=[("PBIX files", "*.pbix"), ("All files", "*.*")]
        )
        if path:
            self.master_var.set(path)
            self.output_dir = os.path.dirname(path)

    def pick_reports_folder(self):
        path = filedialog.askdirectory(title="Select folder with sub reports")
        if path:
            self.reports_var.set(path)

    def run_scan(self):
        master = self.master_var.get().strip()
        reports = self.reports_var.get().strip()

        if not master or not os.path.isfile(master):
            messagebox.showerror("Fejl", "Vælg en gyldig master PBIX.")
            return
        if not reports or not os.path.isdir(reports):
            messagebox.showerror("Fejl", "Vælg en gyldig reports-folder.")
            return

        try:
            cmd = [PYTHON_EXE, SCAN_SCRIPT, master, reports]
            subprocess.Popen(cmd, cwd=BASE_DIR)
            self.status_var.set("Scan.py started. Når den er færdig: klik 'Load Results'.")
        except Exception as e:
            self.status_var.set(f"Fejl: {e}")
            messagebox.showerror("Fejl", f"Kunne ikke starte Scan.py:\n{e}")

    def load_results(self):
        if not self.output_dir:
            master = self.master_var.get().strip()
            if not master:
                messagebox.showerror("Fejl", "Ingen master-PBIX valgt.")
                return
            self.output_dir = os.path.dirname(master)

        try:
            self.data.load(self.output_dir)
            self.status_var.set(f"Data loaded from: {self.output_dir}")
        except Exception as e:
            self.status_var.set(f"Fejl ved load: {e}")
            messagebox.showerror("Fejl", str(e))
            return

        # rebuild dax deps
        self.build_dax_dependencies()

        self.refresh_tables_tab()
        self.refresh_reports_tab()
        self.refresh_relations_tab()
        self.refresh_dax_tab()

    # ---------- TABLES TAB ----------

    def build_tables_tab(self):
        frame = self.tab_tables
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        left = tb.Frame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        right = tb.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tb.Label(left, text="Tables & objects").pack(anchor="w")

        cols = ("type", "used")
        self.tables_tree = tb.Treeview(
            left,
            columns=cols,
            show="tree headings",
            bootstyle="primary"
        )
        self.tables_tree.heading("#0", text="Name")
        self.tables_tree.heading("type", text="Type")
        self.tables_tree.heading("used", text="Used")
        self.tables_tree.column("#0", width=250, anchor="w")
        self.tables_tree.column("type", width=80, anchor="center")
        self.tables_tree.column("used", width=60, anchor="center")

        # farver
        self.tables_tree.tag_configure("used_yes", foreground="lime")
        self.tables_tree.tag_configure("used_no", foreground="red")

        self.tables_tree.pack(fill=tk.BOTH, expand=True)
        self.tables_tree.bind("<<TreeviewSelect>>", self.on_table_tree_select)

        self.tables_tree_map = {}

        tb.Label(right, text="Details").pack(anchor="w")
        self.table_detail_text = tk.Text(right, wrap="word")
        self.table_detail_text.pack(fill=tk.BOTH, expand=True)

    def refresh_tables_tab(self):
        for i in self.tables_tree.get_children():
            self.tables_tree.delete(i)
        self.tables_tree_map.clear()
        self.table_detail_text.delete("1.0", tk.END)

        if self.data.df_usage is None:
            return

        df = self.data.df_usage
        tables = sorted(self.data.tables.keys())
        for t in tables:
            rows = self.data.tables[t]["objects"]
            df_sub = df.iloc[rows]
            used_any_table = "Yes" if (df_sub["UsedAny"] == "Yes").any() else "No"
            tag = "used_yes" if used_any_table == "Yes" else "used_no"

            parent_id = self.tables_tree.insert(
                "",
                "end",
                text=t,
                values=("Table", used_any_table),
                open=False,
                tags=(tag,)
            )
            self.tables_tree_map[parent_id] = (t, None)

            for idx in rows:
                r = df.iloc[idx]
                obj_type = r.get("ObjectType", "")
                name = r.get("Name", "")
                used_any = r.get("UsedAny", "")
                tag_child = "used_yes" if used_any == "Yes" else "used_no"
                text = name if name else f"[{obj_type}]"
                child_id = self.tables_tree.insert(
                    parent_id,
                    "end",
                    text=text,
                    values=(obj_type, used_any),
                    tags=(tag_child,)
                )
                self.tables_tree_map[child_id] = (t, idx)

    def on_table_tree_select(self, event=None):
        sel = self.tables_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        table_name, idx = self.tables_tree_map.get(item_id, (None, None))

        self.table_detail_text.delete("1.0", tk.END)

        if table_name is None:
            return

        if idx is None:
            self.show_table_details(table_name)
        else:
            self.show_object_details(idx)

    def show_table_details(self, table_name: str):
        df = self.data.df_usage
        rows = self.data.tables.get(table_name, {}).get("objects", [])

        text = []
        text.append(f"TABLE: {table_name}")
        text.append("")

        df_sub = df.iloc[rows]
        used_any = "Yes" if (df_sub["UsedAny"] == "Yes").any() else "No"
        text.append(f"Used (any): {used_any}")
        text.append(f"Object count: {len(rows)}")
        text.append("")

        text.append("Objects:")
        for idx in rows:
            r = df.iloc[idx]
            obj_type = r.get("ObjectType", "")
            name = r.get("Name", "")
            used_any = r.get("UsedAny", "")
            text.append(f"  - {obj_type} {name} (UsedAny={used_any})")

        self.table_detail_text.insert("1.0", "\n".join(text))

    def show_object_details(self, idx: int):
        df = self.data.df_usage
        r = df.iloc[idx]

        text = []
        text.append(f"ObjectType : {r.get('ObjectType', '')}")
        text.append(f"Table      : {r.get('Table', '')}")
        text.append(f"Name       : {r.get('Name', '')}")
        text.append(f"FullRef    : {r.get('FullRef', '')}")
        text.append(f"Used       : {r.get('Used', '')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive', '')}")
        text.append(f"UsedAny    : {r.get('UsedAny', '')}")
        text.append(f"UsedInFiles: {r.get('UsedInFiles', '')}")
        text.append("")
        expr = r.get("Expression", "")
        text.append("Expression:")
        if expr:
            text.append(expr)
        else:
            text.append("(none)")

        self.table_detail_text.insert("1.0", "\n".join(text))

    # ---------- REPORTS TAB ----------

    def build_reports_tab(self):
        frame = self.tab_reports
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(1, weight=1)

        search_frame = tb.Frame(frame, padding=5)
        search_frame.grid(row=0, column=0, columnspan=2, sticky="we")
        tb.Label(search_frame, text="Search (reports / table / type / name):").pack(side=tk.LEFT)
        self.report_search_var = tk.StringVar()
        entry = tb.Entry(search_frame, textvariable=self.report_search_var, width=40)
        entry.pack(side=tk.LEFT, padx=5)
        entry.bind("<Return>", self.on_report_search)
        tb.Button(search_frame, text="Search", bootstyle=PRIMARY, command=self.on_report_search).pack(side=tk.LEFT, padx=5)
        tb.Button(search_frame, text="Clear", bootstyle=SECONDARY, command=self.on_report_search_clear).pack(side=tk.LEFT)

        left = tb.Frame(frame, padding=5)
        left.grid(row=1, column=0, sticky="nsew")
        tb.Label(left, text="Reports").pack(anchor="w")
        self.reports_listbox = tk.Listbox(left)
        self.reports_listbox.pack(fill=tk.BOTH, expand=True)
        self.reports_listbox.bind("<<ListboxSelect>>", self.on_report_select)

        right = tb.Frame(frame, padding=5)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        tb.Label(right, text="Objects used in report").grid(row=0, column=0, sticky="w")
        cols = ("type", "table", "name", "usedany")
        self.report_objects_tree = tb.Treeview(
            right,
            columns=cols,
            show="headings",
            bootstyle="primary"
        )
        self.report_objects_tree.heading("type", text="Type")
        self.report_objects_tree.heading("table", text="Table")
        self.report_objects_tree.heading("name", text="Name")
        self.report_objects_tree.heading("usedany", text="UsedAny")
        self.report_objects_tree.column("type", width=80)
        self.report_objects_tree.column("table", width=200)
        self.report_objects_tree.column("name", width=250)
        self.report_objects_tree.column("usedany", width=70, anchor="center")

        self.report_objects_tree.tag_configure("used_yes", foreground="lime")
        self.report_objects_tree.tag_configure("used_no", foreground="red")

        self.report_objects_tree.grid(row=1, column=0, sticky="nsew")
        self.report_objects_tree.bind("<<TreeviewSelect>>", self.on_report_object_select)

        self.report_tree_idx_map = {}

        tb.Label(right, text="Details").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.report_detail_text = tk.Text(right, height=10, wrap="word")
        self.report_detail_text.grid(row=3, column=0, sticky="nsew")

    def refresh_reports_tab(self):
        self.reports_listbox.delete(0, tk.END)
        self.report_objects_tree.delete(*self.report_objects_tree.get_children())
        self.report_tree_idx_map.clear()
        self.report_detail_text.delete("1.0", tk.END)

        if self.data.df_usage is None:
            return

        self._all_report_names = sorted(self.data.reports.keys())
        for name in self._all_report_names:
            self.reports_listbox.insert(tk.END, name)

    def on_report_search(self, event=None):
        query = self.report_search_var.get().strip().lower()
        self.reports_listbox.delete(0, tk.END)

        if self.data.df_usage is None:
            return

        df = self.data.df_usage
        matched_reports = set()

        if not query:
            for name in self._all_report_names:
                self.reports_listbox.insert(tk.END, name)
            return

        for report_name, rows in self.data.reports.items():
            rowtext = report_name.lower()
            for idx in rows:
                r = df.iloc[idx]
                rowtext += " "
                rowtext += str(r.get("Table", "")).lower()
                rowtext += " "
                rowtext += str(r.get("ObjectType", "")).lower()
                rowtext += " "
                rowtext += str(r.get("Name", "")).lower()
            if query in rowtext:
                matched_reports.add(report_name)

        for name in sorted(matched_reports):
            self.reports_listbox.insert(tk.END, name)

    def on_report_search_clear(self):
        self.report_search_var.set("")
        self.refresh_reports_tab()

    def on_report_select(self, event=None):
        sel = self.reports_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        report_name = self.reports_listbox.get(idx)

        self.report_objects_tree.delete(*self.report_objects_tree.get_children())
        self.report_tree_idx_map.clear()
        self.report_detail_text.delete("1.0", tk.END)

        rows = self.data.reports.get(report_name, [])
        df = self.data.df_usage

        for ridx in rows:
            r = df.iloc[ridx]
            obj_type = r.get("ObjectType", "")
            table = r.get("Table", "")
            name = r.get("Name", "")
            used_any = r.get("UsedAny", "")
            tag = "used_yes" if used_any == "Yes" else "used_no"
            item_id = self.report_objects_tree.insert(
                "",
                "end",
                values=(obj_type, table, name, used_any),
                tags=(tag,)
            )
            self.report_tree_idx_map[item_id] = ridx

    def on_report_object_select(self, event=None):
        sel = self.report_objects_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        ridx = self.report_tree_idx_map.get(item_id)
        if ridx is None:
            return

        df = self.data.df_usage
        r = df.iloc[ridx]

        text = []
        text.append(f"Report: {', '.join(str(r.get('UsedInFiles','')).split('|'))}")
        text.append("")
        text.append(f"ObjectType : {r.get('ObjectType', '')}")
        text.append(f"Table      : {r.get('Table', '')}")
        text.append(f"Name       : {r.get('Name', '')}")
        text.append(f"FullRef    : {r.get('FullRef', '')}")
        text.append(f"Used       : {r.get('Used', '')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive', '')}")
        text.append(f"UsedAny    : {r.get('UsedAny', '')}")
        text.append("")
        expr = r.get("Expression", "")
        text.append("Expression:")
        if expr:
            text.append(expr)
        else:
            text.append("(none)")

        self.report_detail_text.delete("1.0", tk.END)
        self.report_detail_text.insert("1.0", "\n".join(text))

    # ---------- RELATIONS TAB ----------

    def build_relations_tab(self):
        frame = self.tab_rel
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.rowconfigure(0, weight=1)

        left = tb.Frame(frame, padding=5)
        mid = tb.Frame(frame, padding=5)
        right = tb.Frame(frame, padding=5)

        left.grid(row=0, column=0, sticky="nsew")
        mid.grid(row=0, column=1, sticky="nsew")
        right.grid(row=0, column=2, sticky="nsew")

        # LEFT: tables
        tb.Label(left, text="Tables").pack(anchor="w")
        self.rel_table_list = tk.Listbox(left)
        self.rel_table_list.pack(fill=tk.BOTH, expand=True)
        self.rel_table_list.bind("<<ListboxSelect>>", self.on_rel_table_select)

        # MIDDLE: local columns (in selected table)
        tb.Label(mid, text="Columns with relations").pack(anchor="w")
        cols_mid = ("column", "direction")
        self.rel_local_tree = tb.Treeview(
            mid,
            columns=cols_mid,
            show="headings",
            bootstyle="primary"
        )
        self.rel_local_tree.heading("column", text="Column")
        self.rel_local_tree.heading("direction", text="Dir")
        self.rel_local_tree.column("column", width=200)
        self.rel_local_tree.column("direction", width=60, anchor="center")
        self.rel_local_tree.pack(fill=tk.BOTH, expand=True)
        self.rel_local_tree.bind("<<TreeviewSelect>>", self.on_rel_local_select)

        # RIGHT: other side
        tb.Label(right, text="Related columns").pack(anchor="w")
        cols_right = ("table", "column", "AnyEndUsed", "UsedInFiles")
        self.rel_other_tree = tb.Treeview(
            right,
            columns=cols_right,
            show="headings",
            bootstyle="primary"
        )
        for c in cols_right:
            self.rel_other_tree.heading(c, text=c)
        self.rel_other_tree.column("table", width=180)
        self.rel_other_tree.column("column", width=180)
        self.rel_other_tree.column("AnyEndUsed", width=80, anchor="center")
        self.rel_other_tree.column("UsedInFiles", width=220)
        self.rel_other_tree.pack(fill=tk.BOTH, expand=True)

        self.rel_tables = []
        self.rel_local_map = {}      # column -> {"dirs": set, "rows": [idx]}
        self.rel_local_item_map = {} # treeitem -> column

    def refresh_relations_tab(self):
        self.rel_table_list.delete(0, tk.END)
        self.rel_local_tree.delete(*self.rel_local_tree.get_children())
        self.rel_other_tree.delete(*self.rel_other_tree.get_children())
        self.rel_local_map.clear()
        self.rel_local_item_map.clear()

        if self.data.df_rel is None:
            return

        df = self.data.df_rel
        tables = set()
        for _, r in df.iterrows():
            ft = str(r.get("FromTableName", "") or "")
            tt = str(r.get("ToTableName", "") or "")
            if ft:
                tables.add(ft)
            if tt:
                tables.add(tt)
        self.rel_tables = sorted(tables)

        for t in self.rel_tables:
            self.rel_table_list.insert(tk.END, t)

    def on_rel_table_select(self, event=None):
        sel = self.rel_table_list.curselection()
        if not sel:
            return
        idx = sel[0]
        table = self.rel_table_list.get(idx)

        self.rel_local_tree.delete(*self.rel_local_tree.get_children())
        self.rel_other_tree.delete(*self.rel_other_tree.get_children())
        self.rel_local_map.clear()
        self.rel_local_item_map.clear()

        df = self.data.df_rel

        # find relations with this table in either end
        for i, r in df.iterrows():
            ft = str(r.get("FromTableName", "") or "")
            fc = str(r.get("FromColumnName", "") or "")
            tt = str(r.get("ToTableName", "") or "")
            tc = str(r.get("ToColumnName", "") or "")

            if ft == table:
                d = self.rel_local_map.setdefault(fc, {"dirs": set(), "rows": []})
                d["dirs"].add("From")
                d["rows"].append(i)
            if tt == table:
                d = self.rel_local_map.setdefault(tc, {"dirs": set(), "rows": []})
                d["dirs"].add("To")
                d["rows"].append(i)

        for col, info in sorted(self.rel_local_map.items(), key=lambda x: x[0]):
            dirs = info["dirs"]
            if "From" in dirs and "To" in dirs:
                direction = "Both"
            elif "From" in dirs:
                direction = "From"
            elif "To" in dirs:
                direction = "To"
            else:
                direction = ""
            item_id = self.rel_local_tree.insert("", "end", values=(col, direction))
            self.rel_local_item_map[item_id] = col

    def on_rel_local_select(self, event=None):
        sel = self.rel_local_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        col = self.rel_local_item_map.get(item_id)
        if col is None:
            return

        self.rel_other_tree.delete(*self.rel_other_tree.get_children())

        df = self.data.df_rel

        # find selected table
        sel_table_idx = self.rel_table_list.curselection()
        if not sel_table_idx:
            return
        table = self.rel_table_list.get(sel_table_idx[0])

        info = self.rel_local_map.get(col, {})
        rows = info.get("rows", [])

        seen = set()
        for i in rows:
            r = df.iloc[i]
            ft = str(r.get("FromTableName", "") or "")
            fc = str(r.get("FromColumnName", "") or "")
            tt = str(r.get("ToTableName", "") or "")
            tc = str(r.get("ToColumnName", "") or "")
            any_used = str(r.get("AnyEndUsed", "") or "")
            used_files = str(r.get("UsedInFiles", "") or "")

            if ft == table and fc == col:
                other_table, other_col = tt, tc
            elif tt == table and tc == col:
                other_table, other_col = ft, fc
            else:
                continue

            key = (other_table, other_col, any_used, used_files)
            if key in seen:
                continue
            seen.add(key)

            self.rel_other_tree.insert(
                "",
                "end",
                values=(other_table, other_col, any_used, used_files)
            )

    # ---------- DAX TAB ----------

    def build_dax_tab(self):
        frame = self.tab_dax
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.columnconfigure(0, weight=1)

        search_frame = tb.Frame(frame, padding=5)
        search_frame.grid(row=0, column=0, sticky="we")
        tb.Label(search_frame, text="Search in DAX (table / name / expression):").pack(side=tk.LEFT)
        self.dax_search_var = tk.StringVar()
        entry = tb.Entry(search_frame, textvariable=self.dax_search_var, width=40)
        entry.pack(side=tk.LEFT, padx=5)
        entry.bind("<Return>", self.on_dax_search)
        tb.Button(search_frame, text="Search", bootstyle=PRIMARY, command=self.on_dax_search).pack(side=tk.LEFT, padx=5)
        tb.Button(search_frame, text="Clear", bootstyle=SECONDARY, command=self.on_dax_search_clear).pack(side=tk.LEFT)

        tb.Label(frame, text="DAX objects").grid(row=1, column=0, sticky="w", padx=5)
        cols = ("ObjectType", "Table", "Name", "UsedAny")
        self.dax_tree = tb.Treeview(
            frame,
            columns=cols,
            show="headings",
            bootstyle="primary"
        )
        for c in cols:
            self.dax_tree.heading(c, text=c)
        self.dax_tree.column("ObjectType", width=100)
        self.dax_tree.column("Table", width=180)
        self.dax_tree.column("Name", width=220)
        self.dax_tree.column("UsedAny", width=80, anchor="center")

        self.dax_tree.tag_configure("used_yes", foreground="lime")
        self.dax_tree.tag_configure("used_no", foreground="red")

        self.dax_tree.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 5))
        self.dax_tree.bind("<<TreeviewSelect>>", self.on_dax_select)

        tb.Label(frame, text="DAX details & dependencies").grid(row=3, column=0, sticky="w", padx=5)
        self.dax_detail_text = tk.Text(frame, height=10, wrap="word")
        self.dax_detail_text.grid(row=4, column=0, sticky="nsew", padx=5, pady=(0, 5))

        self.dax_tree_idx_map = {}

    def build_dax_dependencies(self):
        """
        Bygger en deps-mapping for hele df_usage:
        idx -> set(indices) som denne DAX bruger direkte.
        """
        self.dax_deps = {}
        if self.data.df_usage is None:
            return

        df = self.data.df_usage.fillna("")

        n = len(df)
        table_name_map = {}
        global_name_map = {}

        for i, row in df.iterrows():
            name = str(row.get("Name", "") or "").strip()
            table = str(row.get("Table", "") or "").strip()
            if not name:
                continue
            name_l = name.lower()
            table_l = table.lower()

            table_name_map.setdefault((table_l, name_l), set()).add(i)
            global_name_map.setdefault(name_l, set()).add(i)

        bracket = re.compile(r"\[([^\]]+)\]")

        for i, row in df.iterrows():
            expr = str(row.get("Expression", "") or "")
            if not expr:
                continue
            expr_low = expr.lower()
            table_i = str(row.get("Table", "") or "").strip().lower()

            deps = set()
            for m in bracket.finditer(expr_low):
                token = m.group(1).strip().lower()
                if not token:
                    continue
                idxs = table_name_map.get((table_i, token), set())
                if not idxs:
                    idxs = global_name_map.get(token, set())
                for j in idxs:
                    if j != i:
                        deps.add(j)
            if deps:
                self.dax_deps[i] = deps

    def refresh_dax_tab(self):
        self.dax_tree.delete(*self.dax_tree.get_children())
        self.dax_detail_text.delete("1.0", tk.END)
        self.dax_tree_idx_map.clear()

        if self.data.df_usage is None:
            return

        df = self.data.df_usage
        df_dax = df[df["ObjectType"].isin(["Measure", "CalcColumn", "CalcTable"])]

        for idx, r in df_dax.iterrows():
            used_any = r.get("UsedAny", "")
            tag = "used_yes" if used_any == "Yes" else "used_no"
            item_id = self.dax_tree.insert(
                "",
                "end",
                values=(
                    r.get("ObjectType", ""),
                    r.get("Table", ""),
                    r.get("Name", ""),
                    used_any,
                ),
                tags=(tag,)
            )
            self.dax_tree_idx_map[item_id] = idx

    def on_dax_search(self, event=None):
        query = self.dax_search_var.get().strip().lower()
        self.dax_tree.delete(*self.dax_tree.get_children())
        self.dax_tree_idx_map.clear()

        if self.data.df_usage is None:
            return

        df = self.data.df_usage
        df_dax = df[df["ObjectType"].isin(["Measure", "CalcColumn", "CalcTable"])]

        if not query:
            for idx, r in df_dax.iterrows():
                used_any = r.get("UsedAny", "")
                tag = "used_yes" if used_any == "Yes" else "used_no"
                item_id = self.dax_tree.insert(
                    "",
                    "end",
                    values=(
                        r.get("ObjectType", ""),
                        r.get("Table", ""),
                        r.get("Name", ""),
                        used_any,
                    ),
                    tags=(tag,)
                )
                self.dax_tree_idx_map[item_id] = idx
            return

        for idx, r in df_dax.iterrows():
            rowtext = " ".join([
                str(r.get("Table", "")).lower(),
                str(r.get("Name", "")).lower(),
                str(r.get("Expression", "")).lower(),
            ])
            if query in rowtext:
                used_any = r.get("UsedAny", "")
                tag = "used_yes" if used_any == "Yes" else "used_no"
                item_id = self.dax_tree.insert(
                    "",
                    "end",
                    values=(
                        r.get("ObjectType", ""),
                        r.get("Table", ""),
                        r.get("Name", ""),
                        used_any,
                    ),
                    tags=(tag,)
                )
                self.dax_tree_idx_map[item_id] = idx

    def on_dax_search_clear(self):
        self.dax_search_var.set("")
        self.refresh_dax_tab()

    def on_dax_select(self, event=None):
        sel = self.dax_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        idx = self.dax_tree_idx_map.get(item_id)
        if idx is None:
            return

        df = self.data.df_usage
        r = df.iloc[idx]

        text = []
        text.append(f"ObjectType : {r.get('ObjectType', '')}")
        text.append(f"Table      : {r.get('Table', '')}")
        text.append(f"Name       : {r.get('Name', '')}")
        text.append(f"FullRef    : {r.get('FullRef', '')}")
        text.append(f"Used       : {r.get('Used', '')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive', '')}")
        text.append(f"UsedAny    : {r.get('UsedAny', '')}")
        text.append(f"UsedInFiles: {r.get('UsedInFiles', '')}")
        text.append("")
        text.append("Expression:")
        expr = r.get("Expression", "")
        if expr:
            text.append(expr)
        else:
            text.append("(none)")
        text.append("")

        # dependencies: hvad trækker denne DAX på?
        deps = self.dax_deps.get(idx, set())
        text.append("Depends on:")
        if not deps:
            text.append("  (no referenced DAX objects found via [..] tokens)")
        else:
            dep_rows = [df.iloc[d] for d in sorted(deps)]
            for dr in dep_rows:
                text.append(
                    f"  - {dr.get('ObjectType','')} "
                    f"{dr.get('Table','')}[{dr.get('Name','')}] "
                    f"(UsedAny={dr.get('UsedAny','')})"
                )

        self.dax_detail_text.delete("1.0", tk.END)
        self.dax_detail_text.insert("1.0", "\n".join(text))


def main():
    app = tb.Window(themename="cyborg")
    ViewerGUI(app)
    app.mainloop()


if __name__ == "__main__":
    main()
