PowerBiNinja - The Ultimate Power BI Cleanup & Migration Scanner
---------------------------------------------------------------

Fast. Accurate. Enterprise-grade.

Why PowerBiNinja Exists
-----------------------
I built PowerBiNinja because I needed to move from one Active Directory to another — 
and before migrating dozens of PBIX files, I wanted to know:

    “What should I actually bring — and what can I safely delete?”

Power BI offers no tool for this.
Power BI Helper is slow and incomplete.
Microsoft’s metadata isn’t deep enough.

So PowerBiNinja was created as a full cleanup assistant, capable of scanning a master dataset 
and all thin reports connected to it, then telling you:

- What is used
- What is unused
- What is safe to delete
- Where every DAX object is referenced
- What visuals use what fields
- Which relationships matter
- Which fields are dead, redundant, or legacy garbage
- if you have a big moddel you know the pain of trying to find all relationships - now its Very easy
  - Click on a table and see Everything it connects (to/from) and on what fields.

It is now a full dependency scanner + cleanup engine for any Power BI ecosystem.

What It Does
------------
- Scan a master dataset + all thin reports (extremely fast)
- Extract ALL metadata (tables, columns, measures, relationships, visuals, roles, DAX)
- Detect unused objects (columns, measures, tables, relationships)
- Detect transitive DAX usage (indirect dependencies)
- Map usage back to report > page > visual > field
- Export everything to CSV + Excel
- Full DAX extraction with dependency analysis

Perfect for
-----------
- Cleanup
- Migration to new AD, workspace, or dataset
- Refactoring and governance
- Understanding complex models

GUI Features
------------
- Tables tab with usage color indicators
- Reports tab with hierarchical Report -> Tables -> Fields
- Relations tab with both-end relationship mapping
- DAX tab with expressions and dependency tracing

Installation
------------
pip install -r requirements.txt
python Viewer.py

Inputs
------
- Master PBIX file
- Folder containing thin reports

Outputs
-------
- model_usage.csv
- model_relationships.csv
- pbix_structure.csv
- model_overview.xlsx

Privacy
-------
No data is loaded.
No queries run.
Only metadata is extracted.
Everything stays local.
unlike tools like PowerBiHelper you dont even neeed powerbi Deesktop

Summary
-------
PowerBiNinja is a cleanup and migration tool for Power BI models and thin reports.
It finds what’s used, what’s unused, and everything in between — fast.
If Power BI Helper were rebuilt today for enterprises, this would be it.
