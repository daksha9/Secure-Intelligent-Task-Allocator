"""
dataset_generator.py
Generates the 2,000-record synthetic training dataset (task_dataset.csv)
for the Secure Intelligent Task Scheduling system.

Label rules (applied in priority order):
  1. priority == 3 AND security == 3  →  VM-2
  2. cpu > 80                         →  VM-2
  3. memory > 4000                    →  VM-1
  4. otherwise                        →  VM-0

Run:
    python dataset_generator.py
Output:
    task_dataset.csv  (in the same directory)
"""

import random
import pandas as pd

random.seed(42)

TOTAL_RECORDS = 2000

def assign_label(cpu, memory, priority, security):
    if priority == 3 and security == 3:
        return "VM-2"
    if cpu > 80:
        return "VM-2"
    if memory > 4000:
        return "VM-1"
    return "VM-0"

records = []
for i in range(TOTAL_RECORDS):
    cpu      = random.randint(10, 100)
    memory   = random.randint(512, 8192)
    priority = random.randint(1, 3)
    security = random.randint(1, 3)
    label    = assign_label(cpu, memory, priority, security)
    records.append({
        "task":     f"Task-{i + 1:04d}",
        "cpu":      cpu,
        "memory":   memory,
        "priority": priority,
        "security": security,
        "vm":       label,
    })

df = pd.DataFrame(records)
df.to_csv("task_dataset.csv", index=False)

# Quick distribution summary
dist = df["vm"].value_counts().to_dict()
print(f"Dataset generated: task_dataset.csv  ({TOTAL_RECORDS} records)")
print(f"  VM-0: {dist.get('VM-0', 0)}  VM-1: {dist.get('VM-1', 0)}  VM-2: {dist.get('VM-2', 0)}")
