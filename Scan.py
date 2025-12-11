
import os
import csv
import json
import zipfile
import sys
import re
from collections import defaultdict, deque
import pandas as pd

from pbixray import PBIXRay

# ========= INPUT & PATH CONFIG =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 2:
    MASTER_PBIX_PATH = sys.argv[1]
else:
    MASTER_PBIX_PATH = ""

if len(sys.argv) >= 3:
    REPORTS_FOLDER = sys.argv[2]
else:
    REPORTS_FOLDER = ""

if MASTER_PBIX_PATH:
    OUTPUT_DIR = os.path.dirname(MASTER_PBIX_PATH)
else:
    OUTPUT_DIR = BASE_DIR

PBIX_STRUCTURE_CSV    = os.path.join(OUTPUT_DIR, "pbix_structure.csv")
MODEL_USAGE_CSV       = os.path.join(OUTPUT_DIR, "model_usage.csv")
RELATIONSHIPS_CSV     = os.path.join(OUTPUT_DIR, "model_relationships.csv")
REPORT_USAGE_CSV      = os.path.join(OUTPUT_DIR, "report_usage_details.csv")
OVERVIEW_XLSX         = os.path.join(OUTPUT_DIR, "model_overview.xlsx")

# ========= FILTERS =========
EXCLUDE_PREFIXES = ["LocalDateTable_", "DateTableTemplate_"]
EXCLUDE_EXACT = []
EXCLUDE_CONTAINS = []

def is_meta_table(name: str) -> bool:
    if name in EXCLUDE_EXACT:
        return True
    for p in EXCLUDE_PREFIXES:
        if name.startswith(p):
            return True
    for c in EXCLUDE_CONTAINS:
        if c in name:
            return True
    return False

def load_master_model(master_path: str):
    print(f"Loader master-model: {master_path}")
    model = PBIXRay(master_path)

    schema         = model.schema
    dax_measures   = model.dax_measures
    dax_columns    = model.dax_columns
    dax_tables     = model.dax_tables
    relationships  = model.relationships

    objects = []

    # Tables
    for t in sorted(set(schema['TableName'].tolist())):
        if is_meta_table(t):
            continue
        objects.append({
            "object_type": "Table",
            "table": t,
            "name": "",
            "full_ref": t,
            "expression": "",
            "pattern": t.lower(),
            "used_in": set(),
            "used_locations": [],
        })

    # Columns
    for _, row in schema.iterrows():
        t = row["TableName"]
        if is_meta_table(t):
            continue
        c = row["ColumnName"]
        objects.append({
            "object_type": "Column",
            "table": t,
            "name": c,
            "full_ref": f"{t}[{c}]",
            "expression": "",
            "pattern": c.lower(),
            "used_in": set(),
            "used_locations": [],
        })

    # Measures
    if not dax_measures.empty:
        for _, row in dax_measures.iterrows():
            t = row["TableName"]
            if is_meta_table(t):
                continue
            name = row["Name"]
            expr = row.get("Expression", "")
            objects.append({
                "object_type": "Measure",
                "table": t,
                "name": name,
                "full_ref": f"{t}[{name}]",
                "expression": expr,
                "pattern": name.lower(),
                "used_in": set(),
                "used_locations": [],
            })

    # Calc Columns
    if not dax_columns.empty:
        for _, row in dax_columns.iterrows():
            t = row["TableName"]
            if is_meta_table(t):
                continue
            name = row["ColumnName"]
            expr = row.get("Expression", "")
            objects.append({
                "object_type": "CalcColumn",
                "table": t,
                "name": name,
                "full_ref": f"{t}[{name}]",
                "expression": expr,
                "pattern": name.lower(),
                "used_in": set(),
                "used_locations": [],
            })

    # Calc Tables
    if not dax_tables.empty:
        for _, row in dax_tables.iterrows():
            t = row["TableName"]
            if is_meta_table(t):
                continue
            expr = row.get("Expression", "")
            objects.append({
                "object_type": "CalcTable",
                "table": t,
                "name": "",
                "full_ref": t,
                "expression": expr,
                "pattern": t.lower(),
                "used_in": set(),
                "used_locations": [],
            })

    return model, schema, objects, relationships

# ========= PBIX Layout reading =========
def _decode_layout_bytes(raw: bytes) -> str:
    for enc in ('utf-16-le', 'utf-8', 'cp1252'):
        try:
            t = raw.decode(enc)
            break
        except UnicodeDecodeError:
            t = None
    if t is None:
        return ""
    # Clean known control chars
    t = t.replace('\ufeff', '').replace('\x00', '')
    for ch in (chr(28), chr(29), chr(25)):
        t = t.replace(ch, '')
    # Remove other ASCII control chars except \t\r\n
    t = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', t)
    return t

def read_layout_text_from_pbix(pbix_path: str) -> str:
    try:
        with zipfile.ZipFile(pbix_path, 'r') as zf:
            if 'Report/Layout' not in zf.namelist():
                return ''
            raw = zf.read('Report/Layout')
            text = _decode_layout_bytes(raw)
            return text.lower() if text else ''
    except Exception:
        return ''

def load_layout_json(pbix_path: str):
    try:
        with zipfile.ZipFile(pbix_path, 'r') as zf:
            if 'Report/Layout' not in zf.namelist():
                return None
            raw = zf.read('Report/Layout')
            text = _decode_layout_bytes(raw)
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                return None
    except Exception:
        return None

def iter_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_strings(it)
    elif isinstance(obj, str):
        yield obj

def parse_layout_usage_details(pbix_path: str):
    """
    Parses Report/Layout JSON to extract per-visual info.
    Also returns a lowercased 'blob' per visual of all strings for fallback matching.
    """
    j = load_layout_json(pbix_path)
    if not isinstance(j, dict):
        return []

    sections = j.get('sections') or j.get('Sections') or []
    details = []
    bracket_re = re.compile(r'([A-Za-z0-9_ .\-]+)\[([^\]]+)\]')
    dot_re = re.compile(r'([A-Za-z0-9_ .\-]+)\.([A-Za-z0-9_ .\-]+)')

    for sec in sections:
        page_name = sec.get('displayName') or sec.get('name') or ''
        containers = sec.get('visualContainers') or sec.get('VisualContainers') or []
        for i, vc in enumerate(containers):
            vis_name = vc.get('name') or vc.get('id') or vc.get('parentGroupName') or f"Visual_{i+1}"
            cfg = vc.get('config') or {}
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {}

            single = {}
            if isinstance(cfg, dict):
                single = cfg.get('singleVisual') or cfg.get('visual') or {}

            vis_type = ''
            if isinstance(single, dict):
                vis_type = single.get('visualType') or vc.get('type') or vc.get('visualType') or ''

            refs = []  # (kind, table, name, role, where)
            # 1) projections/queryRef
            projections = {}
            if isinstance(single, dict):
                projections = single.get('projections') or {}
            if isinstance(projections, dict):
                for role, arr in projections.items():
                    if isinstance(arr, list):
                        for ent in arr:
                            if isinstance(ent, dict):
                                qr = ent.get('queryRef')
                                if isinstance(qr, str) and qr:
                                    if '.' in qr:
                                        t, n = qr.split('.', 1)
                                        refs.append(('projection', t.strip(), n.strip(), role, 'projections'))
                                    else:
                                        refs.append(('projection', '', qr.strip(), role, 'projections'))

            # 2) general strings inside cfg
            strings = []
            for s in iter_strings(cfg):
                if isinstance(s, str) and len(s) <= 2000:
                    strings.append(s)
                    for m in bracket_re.finditer(s):
                        refs.append(('bracket', m.group(1).strip(), m.group(2).strip(), '', 'strings'))
                    for m in dot_re.finditer(s):
                        refs.append(('dot', m.group(1).strip(), m.group(2).strip(), '', 'strings'))

            blob = " ".join(strings).lower()

            details.append({
                'page': page_name,
                'visual_id': vis_name,
                'visual_type': vis_type,
                'refs': refs,
                'blob': blob,
            })

    return details

def build_object_indexes(objects):
    table_index = {}
    col_index = {}
    meas_index = {}
    meas_by_name = defaultdict(set)

    for i, o in enumerate(objects):
        ot = o['object_type']
        t = (o['table'] or '').strip().lower()
        n = (o['name'] or '').strip().lower()

        if ot == 'Table' and n == '':
            table_index[t] = i
        elif ot in ('Column', 'CalcColumn'):
            col_index[(t, n)] = i
        elif ot == 'Measure':
            meas_index[(t, n)] = i
            if n:
                meas_by_name[n].add(i)

    return {
        'table_index': table_index,
        'col_index': col_index,
        'meas_index': meas_index,
        'meas_by_name': meas_by_name,
    }

def mark_usage(objects, idx, report_name, page, visual_id, visual_type, role, where, report_usage_rows):
    o = objects[idx]
    o['used_in'].add(report_name)
    o['used_locations'].append({
        'report': report_name,
        'page': page or '',
        'visual_id': visual_id or '',
        'visual_type': visual_type or '',
        'role': role or '',
        'where': where or '',
    })
    report_usage_rows.append({
        'Report': report_name,
        'Page': page or '',
        'VisualId': visual_id or '',
        'VisualType': visual_type or '',
        'Role': role or '',
        'ObjectType': o['object_type'],
        'Table': o['table'],
        'Name': o['name'],
        'FullRef': o['full_ref'],
        'Where': where or '',
    })

def scan_reports_for_usage(objects, reports_folder: str):
    report_usage_rows = []

    if not os.path.isdir(reports_folder):
        print(f"Advarsel: folder findes ikke: {reports_folder}")
        return report_usage_rows

    pbix_files = [
        os.path.join(reports_folder, f)
        for f in os.listdir(reports_folder)
        if f.lower().endswith('.pbix')
    ]

    print(f"Fandt {len(pbix_files)} rapport-filer i {reports_folder}")

    idxs = build_object_indexes(objects)
    table_index = idxs['table_index']
    col_index = idxs['col_index']
    meas_index = idxs['meas_index']
    meas_by_name = idxs['meas_by_name']

    for pbix_path in pbix_files:
        base = os.path.basename(pbix_path)
        print(f"Scanner rapport: {base}")

        # structured details first
        details = parse_layout_usage_details(pbix_path)

        # Fallback broad layout text
        text_lower = read_layout_text_from_pbix(pbix_path) if not details else ""

        # If we have details, try to match via refs
        if details:
            for d in details:
                page = d['page']; visual_id = d['visual_id']; visual_type = d['visual_type']
                for (kind, t, n, role, where) in d['refs']:
                    t_l = (t or '').strip().lower()
                    n_l = (n or '').strip().lower()
                    hit = False
                    if n_l:
                        mi = meas_index.get((t_l, n_l))
                        if mi is not None:
                            mark_usage(objects, mi, base, page, visual_id, visual_type, role, where, report_usage_rows)
                            hit = True
                    if not hit and n_l and (not t_l):
                        for mi in meas_by_name.get(n_l, []):
                            mark_usage(objects, mi, base, page, visual_id, visual_type, role, where, report_usage_rows)
                            hit = True
                    if n_l:
                        ci = col_index.get((t_l, n_l))
                        if ci is not None:
                            mark_usage(objects, ci, base, page, visual_id, visual_type, role, where, report_usage_rows)
                            hit = True

            # Fallback: name-pattern search per visual blob (captures cases where refs not parsed but strings contain)
            for d in details:
                blob = d.get('blob','')
                if not blob:
                    continue
                page = d['page']; visual_id = d['visual_id']; visual_type = d['visual_type']
                for i, o in enumerate(objects):
                    pat = o.get('pattern','')
                    if not pat:
                        continue
                    t = (o['table'] or '').lower()
                    candidates = [
                        pat.lower(),
                        f"{t}[{pat.lower()}]" if t else "",
                        f"{t}.{pat.lower()}" if t else ""
                    ]
                    if any(c and c in blob for c in candidates):
                        mark_usage(objects, i, base, page, visual_id, visual_type, '', 'blob-fallback', report_usage_rows)

        # If we only have broad text, mark UsedInFiles and add report-only location
        if text_lower:
            for i, o in enumerate(objects):
                pat = o['pattern']
                if pat and pat in text_lower:
                    o['used_in'].add(base)
                    # record report-only (page/visual unknown)
                    o['used_locations'].append({
                        'report': base, 'page': '', 'visual_id': '', 'visual_type': '', 'role': '', 'where': 'text-scan'
                    })
                    report_usage_rows.append({
                        'Report': base, 'Page': '', 'VisualId': '', 'VisualType': '',
                        'Role': '', 'ObjectType': o['object_type'], 'Table': o['table'], 'Name': o['name'],
                        'FullRef': o['full_ref'], 'Where': 'text-scan'
                    })

        # Derive table usage from child hits
        used_by_table = defaultdict(bool)
        for i, o in enumerate(objects):
            if base in o['used_in'] and o['table']:
                used_by_table[o['table']] = True
        for t, used in used_by_table.items():
            if used:
                ti = table_index.get(t.lower())
                if ti is not None:
                    objects[ti]['used_in'].add(base)
                    # add derived entries for each page where children were used
                    pages = set()
                    for o in objects:
                        if o['table'] == t:
                            for loc in o.get('used_locations', []):
                                if loc.get('report') == base:
                                    pages.add((loc.get('page',''), loc.get('visual_id',''), loc.get('visual_type','')))
                    if not pages:
                        # report-only
                        objects[ti]['used_locations'].append({
                            'report': base, 'page': '', 'visual_id': '', 'visual_type': '', 'role': '', 'where': 'derived-table'
                        })
                        report_usage_rows.append({
                            'Report': base, 'Page': '', 'VisualId': '', 'VisualType': '',
                            'Role': '', 'ObjectType': 'Table', 'Table': t, 'Name': '', 'FullRef': t, 'Where': 'derived-table'
                        })
                    else:
                        for (pname, vid, vtype) in pages:
                            objects[ti]['used_locations'].append({
                                'report': base, 'page': pname, 'visual_id': vid, 'visual_type': vtype,
                                'role': '', 'where': 'derived-table'
                            })
                            report_usage_rows.append({
                                'Report': base, 'Page': pname, 'VisualId': vid, 'VisualType': vtype,
                                'Role': '', 'ObjectType': 'Table', 'Table': t, 'Name': '', 'FullRef': t, 'Where': 'derived-table'
                            })

    return report_usage_rows

def write_pbix_structure_csv(schema, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Tabel", "Kolonne"])
        for _, r in schema.iterrows():
            if not is_meta_table(r["TableName"]):
                w.writerow([r["TableName"], r["ColumnName"]])
    print(f"Struktur skrevet til: {path}")

def summarize_used_locations(used_locations):
    pages = set()
    visuals = set()
    rpt_expanded = set()
    vtypes = set()
    roles = set()
    for loc in used_locations:
        rep = loc.get('report','')
        pg = loc.get('page','')
        vid = loc.get('visual_id','')
        vt = loc.get('visual_type','')
        role = loc.get('role','')
        if rep and pg:
            pages.add(f"{rep}>{pg}")
        if rep and pg and vid:
            visuals.add(f"{rep}>{pg}>{vid}")
            rpt_expanded.add(f"{rep}>{pg}>{vid}")
        elif rep and pg:
            rpt_expanded.add(f"{rep}>{pg}")
        elif rep:
            # report-only fallback
            rpt_expanded.add(rep)
        if vt:
            vtypes.add(vt)
        if role:
            roles.add(role)
    return {
        'UsedInPages': "|".join(sorted(pages)) if pages else "",
        'UsedInVisuals': "|".join(sorted(visuals)) if visuals else "",
        'UsedInReportsExpanded': "|".join(sorted(rpt_expanded)) if rpt_expanded else "",
        'UsedVisualTypes': "|".join(sorted(vtypes)) if vtypes else "",
        'UsedRoles': "|".join(sorted(roles)) if roles else "",
    }

def write_model_usage_csv(objects, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        header = ["ObjectType", "Table", "Name", "FullRef", "Expression", "Used", "UsedInFiles"]
        header += ["UsedInPages", "UsedInVisuals", "UsedInReportsExpanded", "UsedVisualTypes", "UsedRoles"]
        w.writerow(header)

        for o in objects:
            used = "Yes" if o["used_in"] else "No"
            files = "|".join(sorted(o["used_in"])) if o["used_in"] else ""
            summary = summarize_used_locations(o.get('used_locations', []))
            w.writerow([
                o["object_type"], o["table"], o["name"], o["full_ref"],
                (o["expression"] or "").replace("\n", " ").replace("\r", " "),
                used, files,
                summary['UsedInPages'], summary['UsedInVisuals'], summary['UsedInReportsExpanded'],
                summary['UsedVisualTypes'], summary['UsedRoles']
            ])
    print(f"Model-usage skrevet til: {path}")

def write_relationships_csv(rels, objects, path):
    col_usage = {}
    for o in objects:
        if o["object_type"] in ("Column", "CalcColumn"):
            col_usage.setdefault((o["table"], o["name"]), set()).update(o["used_in"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "FromTableName", "FromColumnName",
            "ToTableName", "ToColumnName",
            "IsActive", "Cardinality", "CrossFilteringBehavior",
            "FromKeyCount", "ToKeyCount", "RelyOnReferentialIntegrity",
            "AnyEndUsed", "UsedInFiles"
        ])

        if rels is not None and not rels.empty:
            for _, r in rels.iterrows():
                ft = r.get("FromTableName", "")
                fc = r.get("FromColumnName", "")
                tt = r.get("ToTableName", "")
                tc = r.get("ToColumnName", "")
                used = set()
                if ft and fc:
                    used.update(col_usage.get((ft, fc), set()))
                if tt and tc:
                    used.update(col_usage.get((tt, tc), set()))
                any_used = "Yes" if used else "No"
                files = "|".join(sorted(used)) if used else ""

                w.writerow([
                    ft, fc, tt, tc,
                    r.get("IsActive", ""),
                    r.get("Cardinality", ""),
                    r.get("CrossFilteringBehavior", ""),
                    r.get("FromKeyCount", ""),
                    r.get("ToKeyCount", ""),
                    r.get("RelyOnReferentialIntegrity", ""),
                    any_used, files
                ])
    print(f"Relationships skrevet til: {path}")

def compute_transitive_used(df_usage):
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

def write_overview_excel(struct_csv, usage_csv, rel_csv, report_usage_csv, excel_path):
    df_struct = pd.read_csv(struct_csv, sep=";")
    df_usage  = pd.read_csv(usage_csv, sep=";").fillna("")
    df_rel    = pd.read_csv(rel_csv, sep=";").fillna("")
    df_rep    = pd.read_csv(report_usage_csv, sep=";").fillna("") if os.path.exists(report_usage_csv) else pd.DataFrame()

    used_trans = compute_transitive_used(df_usage)
    df_usage["UsedTransitive"] = ["Yes" if x else "No" for x in used_trans]

    df_usage_rel = df_usage.copy()
    df_usage_rel["SafeToDelete"] = df_usage_rel["UsedTransitive"].apply(lambda x: "No" if x == "Yes" else "Yes")
    df_delete = df_usage_rel[df_usage_rel["SafeToDelete"] == "Yes"].copy()

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_struct.to_excel(writer, index=False, sheet_name="Structure")
        ws = writer.sheets["Structure"]; rows, cols = df_struct.shape
        ws.add_table(0, 0, rows, cols - 1, {"name": "tblStructure","style": "Table Style Medium 2","columns": [{"header": c} for c in df_struct.columns],})

        df_usage.to_excel(writer, index=False, sheet_name="Usage")
        ws = writer.sheets["Usage"]; rows, cols = df_usage.shape
        ws.add_table(0, 0, rows, cols - 1, {"name": "tblUsage","style": "Table Style Medium 9","columns": [{"header": c} for c in df_usage.columns],})

        df_rel.to_excel(writer, index=False, sheet_name="Relationships")
        ws = writer.sheets["Relationships"]; rows, cols = df_rel.shape
        ws.add_table(0, 0, rows, cols - 1, {"name": "tblRelationships","style": "Table Style Medium 14","columns": [{"header": c} for c in df_rel.columns],})

        df_usage_rel.to_excel(writer, index=False, sheet_name="UsageRel")
        ws = writer.sheets["UsageRel"]; rows, cols = df_usage_rel.shape
        ws.add_table(0, 0, rows, cols - 1, {"name": "tblUsageRel","style": "Table Style Medium 3","columns": [{"header": c} for c in df_usage_rel.columns],})

        df_delete.to_excel(writer, index=False, sheet_name="DeletionCandidates")
        ws = writer.sheets["DeletionCandidates"]; rows, cols = df_delete.shape
        ws.add_table(0, 0, rows, cols - 1, {"name": "tblDelete","style": "Table Style Medium 4","columns": [{"header": c} for c in df_delete.columns],})

        if not df_rep.empty:
            df_rep.to_excel(writer, index=False, sheet_name="ReportUsage")
            ws = writer.sheets["ReportUsage"]; rows, cols = df_rep.shape
            ws.add_table(0, 0, rows, cols - 1, {"name": "tblReportUsage","style": "Table Style Medium 7","columns": [{"header": c} for c in df_rep.columns],})

    print(f"Excel skrevet til: {excel_path}")

def write_report_usage_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        header = ["Report", "Page", "VisualId", "VisualType", "Role", "ObjectType", "Table", "Name", "FullRef", "Where"]
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.get("Report",""), r.get("Page",""), r.get("VisualId",""), r.get("VisualType",""),
                r.get("Role",""), r.get("ObjectType",""), r.get("Table",""), r.get("Name",""),
                r.get("FullRef",""), r.get("Where","")
            ])
    print(f"Report-usage skrevet til: {path}")

def main():
    _, schema, objects, relationships = load_master_model(MASTER_PBIX_PATH)
    report_usage_rows = scan_reports_for_usage(objects, REPORTS_FOLDER)
    write_pbix_structure_csv(schema, PBIX_STRUCTURE_CSV)
    write_model_usage_csv(objects, MODEL_USAGE_CSV)
    write_relationships_csv(relationships, objects, RELATIONSHIPS_CSV)
    write_report_usage_csv(report_usage_rows, REPORT_USAGE_CSV)
    write_overview_excel(PBIX_STRUCTURE_CSV, MODEL_USAGE_CSV, RELATIONSHIPS_CSV, REPORT_USAGE_CSV, OVERVIEW_XLSX)
    print("Færdig.")

if __name__ == "__main__":
    main()
