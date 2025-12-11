# Viewer.py
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict, deque
from pathlib import Path
import re
import pandas as pd

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# =============== CONSTANTS (edit here) ===============
BASE_DIR            = Path(__file__).resolve().parent
PYTHON_EXE          = sys.executable              # use current venv interpreter
SCAN_SCRIPT         = (BASE_DIR / "Scan.py").as_posix()

PBIX_STRUCTURE_CSV  = "pbix_structure.csv"
MODEL_USAGE_CSV     = "model_usage.csv"
RELATIONSHIPS_CSV   = "model_relationships.csv"
REPORT_USAGE_CSV    = "report_usage_details.csv"

COLOR_YES           = "lime"
COLOR_NO            = "red"
# =====================================================


# ---------- helpers ----------
def compute_transitive_used(df_usage: pd.DataFrame):
    df_usage = df_usage.fillna("")
    n = len(df_usage)
    deps = [set() for _ in range(n)]
    table_name_map, global_name_map = {}, {}

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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path.as_posix(), sep=";").fillna("")


def build_report_hierarchy(df_rep: pd.DataFrame):
    """
    rep_h[report][table] = {
        'used': bool,
        'groups': {'Measures': {name:[occ..]}, 'Columns': {...}, 'CalcColumns': {...}, 'CalcTables': {...}},
        'occ': [table-level occ...]
    }
    """
    rep_h = defaultdict(lambda: defaultdict(lambda: {
        "used": False,
        "groups": defaultdict(lambda: defaultdict(list)),
        "occ": []
    }))
    if df_rep.empty:
        return rep_h

    cols = {c.lower(): c for c in df_rep.columns}
    def col(n): return cols.get(n.lower(), n)

    for _, r in df_rep.iterrows():
        rep  = str(r.get(col("Report"), "")) or ""
        page = str(r.get(col("Page"), "")) or ""
        vid  = str(r.get(col("VisualId"), "")) or ""
        vtyp = str(r.get(col("VisualType"), "")) or ""
        role = str(r.get(col("Role"), "")) or ""
        otyp = str(r.get(col("ObjectType"), "")) or ""
        table= str(r.get(col("Table"), "")) or ""
        name = str(r.get(col("Name"), "")) or ""
        where= str(r.get(col("Where"), "")) or ""

        if not rep:
            continue

        if otyp.lower() in ("column", "calccolumn"):
            grp = "Columns"
        elif otyp.lower() == "measure":
            grp = "Measures"
        elif otyp.lower() == "calctable":
            grp = "CalcTables"
        elif otyp.lower() == "table":
            grp = "Table"
        else:
            grp = otyp or "Other"

        occ = {"Page": page, "VisualId": vid, "VisualType": vtyp, "Role": role, "Where": where}
        rep_h[rep][table]["used"] = True
        if grp == "Table":
            rep_h[rep][table]["occ"].append(occ)
        else:
            rep_h[rep][table]["groups"][grp][name].append(occ)
    return rep_h


def fmt_occ_list(occs):
    if not occs:
        return ""
    seen = set()
    out = []
    for o in occs:
        bits = []
        if o.get("Page"): bits.append(o["Page"])
        if o.get("VisualId"): bits.append(o["VisualId"])
        if o.get("VisualType"): bits.append(f"[{o['VisualType']}]")
        if o.get("Role"): bits.append(f"role={o['Role']}")
        if o.get("Where"): bits.append(f"where={o['Where']}")
        s = " > ".join(bits)
        if s and s not in seen:
            seen.add(s); out.append(s)
    return " | ".join(out)


# ---------- Scan dialog ----------
class ScanDialog(tk.Toplevel):
    def __init__(self, parent, master_path: str, reports_folder: str):
        super().__init__(parent)
        self.parent = parent
        self.title("Run Scan")
        self.geometry("780x520")
        self.resizable(True, True)

        self.master_path = master_path
        self.reports_folder = reports_folder

        top = tb.Frame(self, padding=8)
        top.pack(fill=tk.BOTH, expand=True)

        tb.Label(top, text="Master PBIX:").grid(row=0, column=0, sticky="w")
        self.lbl_master = tb.Label(top, text=master_path, bootstyle=INFO)
        self.lbl_master.grid(row=0, column=1, sticky="w")

        tb.Label(top, text="Reports folder:").grid(row=1, column=0, sticky="w", pady=(4,0))
        self.lbl_reports = tb.Label(top, text=reports_folder, bootstyle=INFO)
        self.lbl_reports.grid(row=1, column=1, sticky="w", pady=(4,0))

        tb.Button(top, text="Change folder…", bootstyle=SECONDARY, command=self.change_folder)\
          .grid(row=1, column=2, padx=6, pady=(4,0))

        tb.Label(top, text="Reports to scan:").grid(row=2, column=0, sticky="w", pady=(10,2))
        self.listbox = tk.Listbox(top, height=12)
        self.listbox.grid(row=3, column=0, columnspan=3, sticky="nsew")

        self.pb = tb.Progressbar(top, mode="determinate")
        self.pb.grid(row=4, column=0, columnspan=3, sticky="we", pady=(10,2))

        self.txt = tk.Text(top, height=8, wrap="word")
        self.txt.grid(row=5, column=0, columnspan=3, sticky="nsew")

        btns = tb.Frame(top)
        btns.grid(row=6, column=0, columnspan=3, sticky="e", pady=(10,0))
        self.btn_start = tb.Button(btns, text="Start", bootstyle=SUCCESS, command=self.start_scan)
        self.btn_start.pack(side=tk.RIGHT, padx=4)
        self.btn_cancel = tb.Button(btns, text="Close", bootstyle=SECONDARY, command=self.destroy)
        self.btn_cancel.pack(side=tk.RIGHT, padx=4)

        for c in (0,1,2):
            top.columnconfigure(c, weight=1)
        top.rowconfigure(3, weight=1)
        top.rowconfigure(5, weight=2)

        self.process = None
        self.total_reports = 0
        self.progress_count = 0

        self.refresh_list()

    def change_folder(self):
        path = filedialog.askdirectory(title="Select folder with reports",
                                       initialdir=self.reports_folder or str(BASE_DIR))
        if path:
            self.reports_folder = path.replace("\\", "/")
            self.lbl_reports.config(text=self.reports_folder)
            self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        files = []
        if self.reports_folder and os.path.isdir(self.reports_folder):
            for f in os.listdir(self.reports_folder):
                if f.lower().endswith(".pbix"):
                    files.append(f)
        files.sort(key=str.lower)
        self.total_reports = len(files)
        for f in files:
            self.listbox.insert(tk.END, f)
        self.pb.configure(maximum=max(1, self.total_reports), value=0)
        self.progress_count = 0
        self.log(f"Found {self.total_reports} PBIX in folder.")

    def start_scan(self):
        if not self.master_path or not os.path.isfile(self.master_path):
            messagebox.showerror("Scan", "Invalid master PBIX.")
            return
        if not self.reports_folder or not os.path.isdir(self.reports_folder):
            messagebox.showerror("Scan", "Invalid reports folder.")
            return
        self.btn_start.configure(state="disabled")
        self.pb.configure(value=0)
        self.progress_count = 0
        self.log("Starting Scan.py …")

        try:
            cmd = [PYTHON_EXE, SCAN_SCRIPT, self.master_path, self.reports_folder]
            self.process = subprocess.Popen(
                cmd, cwd=BASE_DIR.as_posix(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            self.after(50, self._poll_stdout)
        except Exception as e:
            messagebox.showerror("Scan", f"Could not start Scan.py:\n{e}")
            self.btn_start.configure(state="normal")

    def _poll_stdout(self):
        if not self.process:
            return
        line = self.process.stdout.readline() if self.process.stdout else ""
        if line:
            self.log(line.rstrip())
            if "Scanner rapport:" in line:
                self.progress_count += 1
                self.pb.configure(value=min(self.progress_count, self.pb.cget("maximum")))
            if "Fandt " in line and " rapport-filer i " in line:
                try:
                    import re as _re
                    num = int(_re.findall(r"Fandt\s+(\d+)\s+rapport-filer", line)[0])
                    self.pb.configure(maximum=max(1, num))
                except Exception:
                    pass
        if self.process.poll() is None:
            self.after(50, self._poll_stdout)
        else:
            rest = self.process.stdout.read() if self.process.stdout else ""
            if rest:
                for ln in rest.splitlines():
                    self.log(ln)
                    if "Scanner rapport:" in ln:
                        self.progress_count += 1
            self.pb.configure(value=self.pb.cget("maximum"))
            self.log("Scan finished.")
            self.btn_start.configure(state="normal")

    def log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)


# ---------- data wrapper ----------
class ModelData:
    def __init__(self):
        self.output_dir = None
        self.df_struct = pd.DataFrame()
        self.df_usage  = pd.DataFrame()
        self.df_rel    = pd.DataFrame()
        self.df_rep    = pd.DataFrame()

        self.tables = {}                         # table -> [row_indices in df_usage]
        self.reports_by_usage = defaultdict(list) # report -> row idx (from UsedInFiles)
        self.rep_h = {}                          # hierarchy from report_usage_details.csv

    def load(self, output_dir: str):
        self.output_dir = output_dir
        struct_path = Path(output_dir, PBIX_STRUCTURE_CSV)
        usage_path  = Path(output_dir, MODEL_USAGE_CSV)
        rel_path    = Path(output_dir, RELATIONSHIPS_CSV)
        rep_path    = Path(output_dir, REPORT_USAGE_CSV)

        self.df_struct = read_csv(struct_path)
        self.df_usage  = read_csv(usage_path)
        self.df_rel    = read_csv(rel_path)
        self.df_rep    = read_csv(rep_path)

        if not self.df_usage.empty:
            used_trans = compute_transitive_used(self.df_usage)
            self.df_usage["UsedTransitive"] = ["Yes" if x else "No" for x in used_trans]
            self.df_usage["UsedAny"] = self.df_usage.apply(
                lambda r: "Yes" if (str(r.get("Used","")).lower()=="yes"
                                    or str(r.get("UsedTransitive","")).lower()=="yes"
                                    or str(r.get("UsedInFiles",""))!="")
                else "No",
                axis=1
            )

        self.tables.clear()
        if not self.df_usage.empty:
            for idx, row in self.df_usage.iterrows():
                t = str(row.get("Table","") or "")
                if t:
                    self.tables.setdefault(t, []).append(idx)

        self.reports_by_usage.clear()
        if not self.df_usage.empty:
            for idx, row in self.df_usage.iterrows():
                s = str(row.get("UsedInFiles","") or "")
                if not s: continue
                for part in s.split("|"):
                    p = part.strip()
                    if p:
                        self.reports_by_usage[p].append(idx)

        self.rep_h = build_report_hierarchy(self.df_rep)


# ---------- GUI ----------
class ViewerGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("PBIX Model Viewer")
        self.root.geometry("1500x900")

        self.data = ModelData()
        self.output_dir = None

        self.report_selected = None
        self.table_selected  = None

        # top bar
        top = tb.Frame(root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        tb.Label(top, text="Master PBIX:").grid(row=0, column=0, sticky="w")
        self.master_var = tk.StringVar()
        tb.Entry(top, textvariable=self.master_var, width=70).grid(row=0, column=1, sticky="we", padx=5)
        tb.Button(top, text="Browse", bootstyle=INFO, command=self.pick_master).grid(row=0, column=2, padx=4)

        tb.Label(top, text="Reports folder:").grid(row=1, column=0, sticky="w", pady=(4,0))
        self.reports_var = tk.StringVar()
        tb.Entry(top, textvariable=self.reports_var, width=70).grid(row=1, column=1, sticky="we", padx=5, pady=(4,0))
        tb.Button(top, text="Browse", bootstyle=INFO, command=self.pick_reports_folder).grid(row=1, column=2, padx=4, pady=(4,0))

        tb.Button(top, text="Run Scan", bootstyle=SUCCESS, width=15, command=self.run_scan).grid(row=0, column=3, padx=(15,5))
        tb.Button(top, text="Load Results", bootstyle=PRIMARY,width=15, command=self.load_results).grid(row=1, column=3, padx=(15,5))

        top.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        tb.Label(root, textvariable=self.status_var, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        # notebook
        self.nb = tb.Notebook(root, bootstyle="primary")
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_tables  = tb.Frame(self.nb)
        self.tab_reports = tb.Frame(self.nb)
        self.tab_rel     = tb.Frame(self.nb)
        self.tab_dax     = tb.Frame(self.nb)

        self.nb.add(self.tab_tables,  text="Tables")
        self.nb.add(self.tab_reports, text="Reports")
        self.nb.add(self.tab_rel,     text="Relations")
        self.nb.add(self.tab_dax,     text="DAX")

        self.build_tables_tab()
        self.build_reports_tab()
        self.build_relations_tab()
        self.build_dax_tab()

    # ---- top actions ----
    def pick_master(self):
        path = filedialog.askopenfilename(title="Select master PBIX", filetypes=[("PBIX","*.pbix"),("All","*.*")])
        if path:
            self.master_var.set(path.replace("\\", "/"))
            self.output_dir = Path(path).parent.as_posix()

    def pick_reports_folder(self):
        path = filedialog.askdirectory(title="Select sub reports folder")
        if path:
            self.reports_var.set(path.replace("\\", "/"))

    def run_scan(self):
        master = self.master_var.get().strip()
        reports = self.reports_var.get().strip()
        if not master or not os.path.isfile(master):
            messagebox.showerror("Fejl", "Vælg en gyldig master PBIX.")
            return
        if not reports or not os.path.isdir(reports):
            messagebox.showerror("Fejl", "Vælg en gyldig reports-folder.")
            return
        dlg = ScanDialog(self.root, master, reports)
        dlg.grab_set()

    def load_results(self):
        if not self.output_dir:
            master = self.master_var.get().strip()
            if not master:
                messagebox.showerror("Fejl", "Ingen master-PBIX valgt.")
                return
            self.output_dir = Path(master).parent.as_posix()

        try:
            self.data.load(self.output_dir)
            self.status_var.set(f"Data loaded from: {self.output_dir}")
        except Exception as e:
            self.status_var.set(f"Fejl ved load: {e}")
            messagebox.showerror("Fejl", str(e))
            return

        self.refresh_tables_tab()
        self.refresh_reports_tab()
        self.refresh_relations_tab()
        self.refresh_dax_tab()

    # ---------------- TABLES TAB ----------------
    def build_tables_tab(self):
        frame = self.tab_tables
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        left = tb.Frame(frame); left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right= tb.Frame(frame); right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tb.Label(left, text="Tables & objects").pack(anchor="w")

        cols = ("type","used")
        self.tables_tree = tb.Treeview(left, columns=cols, show="tree headings", bootstyle="primary")
        self.tables_tree.heading("#0", text="Name")
        self.tables_tree.heading("type", text="Type")
        self.tables_tree.heading("used", text="Used")
        self.tables_tree.column("#0", width=260, anchor="w")
        self.tables_tree.column("type", width=90, anchor="center")
        self.tables_tree.column("used", width=70, anchor="center")
        self.tables_tree.tag_configure("used_yes", foreground=COLOR_YES)
        self.tables_tree.tag_configure("used_no",  foreground=COLOR_NO)
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

        df = self.data.df_usage
        if df.empty:
            return

        tables = sorted(self.data.tables.keys())
        for t in tables:
            rows = self.data.tables[t]
            df_sub = df.iloc[rows]
            used_any_table = "Yes" if (df_sub["UsedAny"]=="Yes").any() else "No"
            tag = "used_yes" if used_any_table=="Yes" else "used_no"

            parent = self.tables_tree.insert("", "end", text=t, values=("Table", used_any_table), tags=(tag,))
            self.tables_tree_map[parent] = (t, None)

            for idx in rows:
                r = df.iloc[idx]
                ot = r.get("ObjectType",""); name = r.get("Name",""); used_any = r.get("UsedAny","")
                tag_child = "used_yes" if used_any=="Yes" else "used_no"
                text = name if name else f"[{ot}]"
                child = self.tables_tree.insert(parent, "end", text=text, values=(ot, used_any), tags=(tag_child,))
                self.tables_tree_map[child] = (t, idx)

    def on_table_tree_select(self, _=None):
        sel = self.tables_tree.selection()
        if not sel: return
        item = sel[0]
        table, idx = self.tables_tree_map.get(item, (None, None))
        self.table_detail_text.delete("1.0", tk.END)
        if table is None: return
        if idx is None:
            self.show_table_details(table)
        else:
            self.table_detail_text.insert("1.0", self.show_object_details(idx, report_filter=""))

    def show_table_details(self, table):
        df = self.data.df_usage
        rows = self.data.tables.get(table, [])
        text = [f"TABLE: {table}", ""]
        df_sub = df.iloc[rows]
        used_any = "Yes" if (df_sub["UsedAny"]=="Yes").any() else "No"
        text.append(f"Used (any): {used_any}")
        text.append(f"Object count: {len(rows)}")
        text.append("")
        text.append("Objects:")
        for idx in rows:
            r = df.iloc[idx]
            text.append(f"  - {r.get('ObjectType','')} {r.get('Name','')} (UsedAny={r.get('UsedAny','')})")
        self.table_detail_text.insert("1.0", "\n".join(text))

    def show_object_details(self, idx: int, report_filter: str):
        df = self.data.df_usage; r = df.iloc[idx]
        text = []
        text.append(f"ObjectType : {r.get('ObjectType','')}")
        text.append(f"Table      : {r.get('Table','')}")
        text.append(f"Name       : {r.get('Name','')}")
        text.append(f"FullRef    : {r.get('FullRef','')}")
        text.append(f"Used       : {r.get('Used','')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive','')}")
        text.append(f"UsedAny    : {r.get('UsedAny','')}")
        text.append(f"UsedInFiles: {r.get('UsedInFiles','')}")
        text.append("")
        expr = r.get("Expression","")
        text.append("Expression:")
        text.append(expr if expr else "(none)")
        return "\n".join(text)

    # ---------------- REPORTS TAB ----------------
    def build_reports_tab(self):
        frame = self.tab_reports
        # Give more space to RIGHT side
        frame.columnconfigure(0, weight=0)   # left tree fixed
        frame.columnconfigure(1, weight=1)   # right expands
        frame.rowconfigure(0, weight=0)
        frame.rowconfigure(1, weight=1)

        # LEFT: report → tables tree
        left = tb.Frame(frame, padding=5)
        left.grid(row=0, column=0, rowspan=2, sticky="nsw")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(2, weight=1)

        tb.Label(left, text="Search (reports / tables / names):").grid(row=0, column=0, sticky="w")
        self.rep_search_var = tk.StringVar()
        e = tb.Entry(left, textvariable=self.rep_search_var)
        e.grid(row=0, column=1, sticky="we", padx=6)
        e.bind("<KeyRelease>", lambda _e: self.refresh_reports_tab())

        tb.Label(left, text="Reports").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,2))
        self.rep_tree_left = tb.Treeview(left, columns=("type",), show="tree headings", bootstyle="primary", height=26)
        self.rep_tree_left.heading("#0", text="Name")
        self.rep_tree_left.heading("type", text="Type")
        self.rep_tree_left.column("#0", width=280, anchor="w")
        self.rep_tree_left.column("type", width=80, anchor="center")
        self.rep_tree_left.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.rep_tree_left.bind("<<TreeviewSelect>>", self.on_rep_left_select)

        self.rep_left_map = {}  # node -> ("report", rep) or ("table", rep, table)

        # RIGHT: objects table (top) + details (bottom)
        right = tb.Frame(frame, padding=5)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=3)   # objects area bigger
        right.rowconfigure(3, weight=1)   # details smaller

        tb.Label(right, text="Objects for selected table: Used in THIS report vs ANY report").grid(row=0, column=0, sticky="w")

        cols = ("type","name","used_this","used_any","pages_visuals")
        self.rep_objects = tb.Treeview(right, columns=cols, show="headings", bootstyle="primary")
        self.rep_objects.heading("type", text="Type")
        self.rep_objects.heading("name", text="Name")
        self.rep_objects.heading("used_this", text="Used (this)")
        self.rep_objects.heading("used_any", text="Used (any)")
        self.rep_objects.heading("pages_visuals", text="Pages/Visuals (this report)")
        self.rep_objects.column("type", width=110, anchor="center")
        self.rep_objects.column("name", width=260, anchor="w")
        self.rep_objects.column("used_this", width=90, anchor="center")
        self.rep_objects.column("used_any", width=90, anchor="center")
        self.rep_objects.column("pages_visuals", width=700, anchor="w")
        self.rep_objects.tag_configure("yes_this", foreground=COLOR_YES)
        self.rep_objects.tag_configure("no_this", foreground=COLOR_NO)
        self.rep_objects.grid(row=1, column=0, sticky="nsew")
        self.rep_objects.bind("<<TreeviewSelect>>", self.on_report_object_select)

        # map: tree item -> df_usage index
        self._rep_obj_idx_map = {}

        tb.Label(right, text="Details").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.rep_detail = tk.Text(right, height=10, wrap="word")
        self.rep_detail.grid(row=3, column=0, sticky="nsew")

    def refresh_reports_tab(self):
        self.rep_tree_left.delete(*self.rep_tree_left.get_children())
        self.rep_left_map.clear()
        self.rep_objects.delete(*self.rep_objects.get_children())
        self._rep_obj_idx_map.clear()
        self.rep_detail.delete("1.0", tk.END)

        df_rep = self.data.df_rep
        df_usage = self.data.df_usage
        rep_h = self.data.rep_h

        reports = set()
        if not df_rep.empty and "Report" in df_rep.columns:
            reports.update(str(x) for x in df_rep["Report"].unique() if str(x))
        elif not df_usage.empty:
            for s in df_usage["UsedInFiles"]:
                if not s: continue
                for part in str(s).split("|"):
                    if part.strip(): reports.add(part.strip())

        reports = sorted(reports, key=lambda x: x.lower())
        q = (self.rep_search_var.get() or "").strip().lower()

        for rep in reports:
            if q and not self._report_matches_query(rep, rep_h.get(rep, {}), q):
                continue
            rep_id = self.rep_tree_left.insert("", "end", text=rep, values=("Report",))
            self.rep_left_map[rep_id] = ("report", rep)
            tables = rep_h.get(rep, {})
            for t in sorted(tables.keys(), key=lambda x: x.lower() if x else ""):
                t_id = self.rep_tree_left.insert(rep_id, "end", text=t, values=("Table",), tags=("used_yes",))
                self.rep_left_map[t_id] = ("table", rep, t)
            self.rep_tree_left.item(rep_id, open=False)

    def _report_matches_query(self, rep: str, info: dict, q: str) -> bool:
        if q in rep.lower():
            return True
        for t, tinf in info.items():
            if t and q in t.lower():
                return True
            for grp, objs in tinf.get("groups", {}).items():
                if grp and q in grp.lower(): return True
                for name, occs in objs.items():
                    if name and q in name.lower(): return True
                    if q in fmt_occ_list(occs).lower(): return True
            if q in fmt_occ_list(tinf.get("occ", [])).lower(): return True
        return False

    def on_rep_left_select(self, _=None):
        sel = self.rep_tree_left.selection()
        if not sel: return
        node = sel[0]
        kind = self.rep_left_map.get(node)
        self.rep_objects.delete(*self.rep_objects.get_children())
        self._rep_obj_idx_map.clear()
        self.rep_detail.delete("1.0", tk.END)
        if not kind: return

        if kind[0] == "report":
            self.report_selected = kind[1]; self.table_selected = None
            self.rep_detail.insert("1.0", f"Selected report: {self.report_selected}\nFold out and select a TABLE to view its objects with usage flags.")
        else:
            rep, table = kind[1], kind[2]
            self.report_selected = rep; self.table_selected = table
            self._fill_report_objects(rep, table)

    def _fill_report_objects(self, report: str, table: str):
        df = self.data.df_usage
        rep_info = self.data.rep_h.get(report, {}).get(table, {})
        occ_map = defaultdict(list)
        if rep_info:
            for grp, objs in rep_info.get("groups", {}).items():
                for name, occs in objs.items():
                    occ_map[(grp, name)].extend(occs)

        rows = self.data.tables.get(table, [])
        # key map for quick lookup from (type,name) -> idx
        key_to_idx = {}
        for idx in rows:
            r = df.iloc[idx]
            key_to_idx[(str(r.get("ObjectType","")).lower(), str(r.get("Name","")))] = idx

        for idx in rows:
            r = df.iloc[idx]
            ot = str(r.get("ObjectType",""))
            name = str(r.get("Name",""))
            used_files = str(r.get("UsedInFiles","") or "")
            used_any = "Yes" if used_files else "No"
            used_this = "No"
            if used_files:
                for part in used_files.split("|"):
                    if part.strip() == report:
                        used_this = "Yes"; break

            if ot.lower()=="measure":
                grp = "Measures"
            elif ot.lower() in ("column","calccolumn"):
                grp = "Columns"
            elif ot.lower()=="calctable":
                grp = "CalcTables"
            else:
                grp = ot

            occ_txt = fmt_occ_list(occ_map.get((grp, name), [])) if name else fmt_occ_list(rep_info.get("occ", []))
            tag = "yes_this" if used_this=="Yes" else "no_this"
            item = self.rep_objects.insert("", "end",
                                           values=(ot, name, used_this, used_any, occ_txt),
                                           tags=(tag,))
            # map item -> df index for details
            self._rep_obj_idx_map[item] = key_to_idx.get((ot.lower(), name), idx)

        self.rep_detail.insert(
            "1.0",
            f"Report: {report}\nTable: {table}\nObjects shown: all from table. "
            f"Columns 'Used (this)'/'Used (any)' reflect usage in this report vs globally."
        )

    def on_report_object_select(self, _=None):
        sel = self.rep_objects.selection()
        if not sel: return
        item = sel[0]
        idx = self._rep_obj_idx_map.get(item)
        if idx is None: return

        df = self.data.df_usage
        r = df.iloc[idx]

        # build occurrence text for THIS report/table if available
        occ_txt = ""
        if self.report_selected and self.table_selected:
            rep_info = self.data.rep_h.get(self.report_selected, {}).get(self.table_selected, {})
            ot = str(r.get("ObjectType",""))
            name = str(r.get("Name",""))
            if ot.lower()=="measure":
                grp = "Measures"
            elif ot.lower() in ("column","calccolumn"):
                grp = "Columns"
            elif ot.lower()=="calctable":
                grp = "CalcTables"
            else:
                grp = ot
            if name:
                occ_txt = fmt_occ_list(rep_info.get("groups", {}).get(grp, {}).get(name, []))
            else:
                occ_txt = fmt_occ_list(rep_info.get("occ", []))

        text = []
        text.append(f"Report     : {self.report_selected or ''}")
        text.append(f"Table      : {r.get('Table','')}")
        text.append(f"ObjectType : {r.get('ObjectType','')}")
        text.append(f"Name       : {r.get('Name','')}")
        text.append(f"FullRef    : {r.get('FullRef','')}")
        text.append(f"Used       : {r.get('Used','')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive','')}")
        text.append(f"UsedAny    : {r.get('UsedAny','')}")
        text.append(f"UsedInFiles: {r.get('UsedInFiles','')}")
        text.append("")
        text.append("Expression:")
        expr = r.get("Expression","")
        text.append(expr if expr else "(none)")
        if occ_txt:
            text.append("")
            text.append("Pages/Visuals in this report:")
            text.append(occ_txt)

        self.rep_detail.delete("1.0", tk.END)
        self.rep_detail.insert("1.0", "\n".join(text))

    # ---------------- RELATIONS TAB ----------------
    def build_relations_tab(self):
        frame = self.tab_rel
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        top = tb.Frame(frame, padding=5)
        top.grid(row=0, column=0, sticky="we")
        tb.Label(top, text="Search:").pack(side=tk.LEFT)
        self.rel_search_var = tk.StringVar()
        ent = tb.Entry(top, textvariable=self.rel_search_var, width=40)
        ent.pack(side=tk.LEFT, padx=6)
        ent.bind("<KeyRelease>", lambda _e: self.refresh_relations_tab())

        cols = ("FromTable","FromColumn","ToTable","ToColumn","Active","Card","Filter","AnyEndUsed","UsedInFiles")
        self.rel_tree = tb.Treeview(frame, columns=cols, show="headings", bootstyle="primary")
        for c, w in zip(cols, (180,160,180,160,70,70,80,90,360)):
            self.rel_tree.heading(c, text=c)
            self.rel_tree.column(c, width=w, anchor="w")
        self.rel_tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def refresh_relations_tab(self):
        self.rel_tree.delete(*self.rel_tree.get_children())
        df = self.data.df_rel
        if df.empty: return
        q = (self.rel_search_var.get() or "").strip().lower()
        for _, r in df.iterrows():
            ft = str(r.get("FromTableName","")); fc = str(r.get("FromColumnName",""))
            tt = str(r.get("ToTableName",""));   tc = str(r.get("ToColumnName",""))
            act = str(r.get("IsActive","")); card = str(r.get("Cardinality",""))
            flt = str(r.get("CrossFilteringBehavior",""))
            anyu= str(r.get("AnyEndUsed","")); files = str(r.get("UsedInFiles",""))
            rowtxt = " ".join([ft,fc,tt,tc,act,card,flt,anyu,files]).lower()
            if q and q not in rowtxt:
                continue
            self.rel_tree.insert("", "end", values=(ft,fc,tt,tc,act,card,flt,anyu,files))

    # ---------------- DAX TAB ----------------
    def build_dax_tab(self):
        frame = self.tab_dax
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        search = tb.Frame(frame, padding=5)
        search.grid(row=0, column=0, sticky="we")
        tb.Label(search, text="Search in DAX (table/name/expression):").pack(side=tk.LEFT)
        self.dax_search_var = tk.StringVar()
        ent = tb.Entry(search, textvariable=self.dax_search_var, width=40)
        ent.pack(side=tk.LEFT, padx=6)
        ent.bind("<KeyRelease>", lambda _e: self.refresh_dax_tab())

        tb.Label(frame, text="DAX objects").grid(row=1, column=0, sticky="w", padx=5)
        cols = ("ObjectType","Table","Name","UsedAny")
        self.dax_tree = tb.Treeview(frame, columns=cols, show="headings", bootstyle="primary")
        for c in cols:
            self.dax_tree.heading(c, text=c)
        self.dax_tree.column("ObjectType", width=120)
        self.dax_tree.column("Table", width=220)
        self.dax_tree.column("Name", width=260)
        self.dax_tree.column("UsedAny", width=90, anchor="center")
        self.dax_tree.tag_configure("used_yes", foreground=COLOR_YES)
        self.dax_tree.tag_configure("used_no",  foreground=COLOR_NO)
        self.dax_tree.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.dax_tree.bind("<<TreeviewSelect>>", self.on_dax_select)

        tb.Label(frame, text="DAX details").grid(row=3, column=0, sticky="w", padx=5)
        self.dax_detail = tk.Text(frame, height=10, wrap="word")
        self.dax_detail.grid(row=4, column=0, sticky="nsew", padx=5, pady=(0,5))

        self._dax_idx_map = {}

    def refresh_dax_tab(self):
        self.dax_tree.delete(*self.dax_tree.get_children())
        self._dax_idx_map.clear()
        df = self.data.df_usage
        if df.empty: return
        q = (self.dax_search_var.get() or "").strip().lower()
        for idx, r in df.iterrows():
            ot = str(r.get("ObjectType","")).lower()
            if ot not in ("measure","calccolumn","calctable"): continue
            rowtxt = " ".join([str(r.get("Table","")), str(r.get("Name","")), str(r.get("Expression",""))]).lower()
            if q and q not in rowtxt:
                continue
            used_any = str(r.get("UsedAny",""))
            tag = "used_yes" if used_any=="Yes" else "used_no"
            item = self.dax_tree.insert("", "end", values=(r.get("ObjectType",""), r.get("Table",""), r.get("Name",""), used_any), tags=(tag,))
            self._dax_idx_map[item] = idx

    def on_dax_select(self, _=None):
        sel = self.dax_tree.selection()
        if not sel: return
        item = sel[0]
        idx = self._dax_idx_map.get(item)
        if idx is None: return
        df = self.data.df_usage; r = df.iloc[idx]
        text = []
        text.append(f"ObjectType : {r.get('ObjectType','')}")
        text.append(f"Table      : {r.get('Table','')}")
        text.append(f"Name       : {r.get('Name','')}")
        text.append(f"FullRef    : {r.get('FullRef','')}")
        text.append(f"Used       : {r.get('Used','')}")
        text.append(f"UsedTrans. : {r.get('UsedTransitive','')}")
        text.append(f"UsedAny    : {r.get('UsedAny','')}")
        text.append(f"UsedInFiles: {r.get('UsedInFiles','')}")
        text.append("")
        expr = r.get("Expression","")
        text.append("Expression:"); text.append(expr if expr else "(none)")
        self.dax_detail.delete("1.0", tk.END)
        self.dax_detail.insert("1.0", "\n".join(text))


def main():
    app = tb.Window(themename="cyborg")
    ViewerGUI(app)
    app.mainloop()


if __name__ == "__main__":
    main()
#2025-12-11 kl 12:12#