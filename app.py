from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import random
import traceback
import os

app = Flask(__name__)

# ── Load model & scaler ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "vm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

ai_model  = load_model(MODEL_PATH)
ai_scaler = joblib.load(SCALER_PATH)

# Fixed VM labels — LabelEncoder sorts alphabetically: VM-0→0, VM-1→1, VM-2→2
VM_LABELS = ["VM-0", "VM-1", "VM-2"]

SCALE_UP_THRESHOLD   = 80
SCALE_DOWN_THRESHOLD = 30
MAX_VMS              = 6
MIN_VMS              = 2

# ── VM personality — each VM has a role & characteristic workload profile ───
# When a new VM is spawned as overflow for a parent, it inherits the parent's
# profile (same priority/security/memory affinity) so tasks are compatible.
VM_PROFILE = {
    "VM-0": {"priority": 1, "security": 1, "memory_max": 2048, "role": "general-purpose baseline"},
    "VM-1": {"priority": 2, "security": 2, "memory_max": 4096, "role": "mid-tier compute"},
    "VM-2": {"priority": 3, "security": 3, "memory_max": 8192, "role": "high-performance"},
}

def get_parent(vm_id):
    """Return the base VM (VM-0/1/2) that a spawned VM belongs to."""
    # VM-3 spawned from VM-0, VM-4 from VM-1, VM-5 from VM-2 etc.
    # Mapping: VM-3→VM-0, VM-4→VM-1, VM-5→VM-2
    base_map = {"VM-3": "VM-0", "VM-4": "VM-1", "VM-5": "VM-2"}
    return base_map.get(vm_id, vm_id)

def get_reason(vm_id, task):
    """Return allocation reason. Overflow VMs explicitly state inherited profile."""
    parent      = get_parent(vm_id)
    is_overflow = vm_id not in VM_LABELS
    profile     = VM_PROFILE.get(parent, VM_PROFILE["VM-0"])

    if parent == "VM-0":
        if task["Security"] == 1 and task["Priority"] == 1:
            r = f"{vm_id}: Low-priority, low-security task → general-purpose baseline VM"
        elif task["Memory"] <= 2048:
            r = f"{vm_id}: General-purpose VM — balanced CPU and memory allocation"
        else:
            r = f"{vm_id}: General-purpose VM — standard workload distribution"

    elif parent == "VM-1":
        if task["CPU"] > 80:
            r = f"{vm_id}: Compute-optimised VM — high CPU demand (Priority≤{profile['priority']}, Security≤{profile['security']})"
        elif task["Memory"] > 512:
            r = f"{vm_id}: Mid-tier VM — memory-intensive task (Memory≤{profile['memory_max']}MB)"
        else:
            r = f"{vm_id}: Mid-tier VM — standard workload (Priority≤{profile['priority']}, Security≤{profile['security']})"

    else:  # VM-2 and its overflow children
        if task["Priority"] == 3:
            r = f"{vm_id}: Performance-tier VM — high-priority task (Priority={task['Priority']}, Security={task['Security']})"
        elif task["Security"] == 3:
            r = f"{vm_id}: Performance-tier VM — high-security workload (Security={task['Security']}, Memory≤{profile['memory_max']}MB)"
        else:
            r = f"{vm_id}: High-performance VM — complex AI/ML workload"

    # For overflow (spawned) VMs, append the inherited profile clearly
    if is_overflow:
        r += (
            f" | Overflow from {parent} — "
            f"inherited profile: Priority≤{profile['priority']}, "
            f"Security≤{profile['security']}, "
            f"Memory≤{profile['memory_max']}MB"
        )
    return r


# ── Route ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():

    results    = []
    vm_count   = {"VM-0": 0, "VM-1": 0, "VM-2": 0}
    vm_load    = {"VM-0": 0, "VM-1": 0, "VM-2": 0}
    vm_summary = []
    vm_health  = {}
    scale_up   = []
    scale_down = []
    pool_size  = 3
    error_msg  = None

    if request.method == "POST":
        try:

            # ── STEP 1: Parse input ─────────────────────────────────────────
            raw_tasks = []
            file_obj  = request.files.get("excel")

            if file_obj and file_obj.filename != "":
                df = pd.read_excel(file_obj)
                df.columns = df.columns.str.strip().str.lower()

                need = {"task", "cpu", "memory", "priority", "security"}
                miss = need - set(df.columns)
                if miss:
                    raise ValueError(
                        f"Missing columns: {miss}. Your file has: {list(df.columns)}"
                    )
                df = df.dropna(subset=list(need))

                # ── Feature range validation ────────────────────────────────
                # Reject rows with out-of-range Priority or Security before
                # they reach the model — out-of-range values cause scaler
                # extrapolation and meaningless predictions.
                invalid_priority = df[(df["priority"] < 1) | (df["priority"] > 3)]
                invalid_security = df[(df["security"] < 1) | (df["security"] > 3)]
                if len(invalid_priority) > 0 or len(invalid_security) > 0:
                    raise ValueError(
                        f"Invalid feature values detected. "
                        f"Priority must be 1–3 (found {len(invalid_priority)} invalid rows). "
                        f"Security must be 1–3 (found {len(invalid_security)} invalid rows). "
                        f"Please correct these values and re-upload."
                    )

                for _, row in df.iterrows():
                    raw_tasks.append({
                        "Task":     str(row["task"]).strip(),
                        "CPU":      int(float(row["cpu"])),
                        "Memory":   int(float(row["memory"])),
                        "Priority": int(float(row["priority"])),
                        "Security": int(float(row["security"])),
                    })
            else:
                names = request.form.getlist("task[]")
                cpus  = request.form.getlist("cpu[]")
                mems  = request.form.getlist("memory[]")
                pris  = request.form.getlist("priority[]")
                secs  = request.form.getlist("security[]")
                for i in range(len(names)):
                    if not names[i].strip():
                        continue
                    pri = int(pris[i])
                    sec = int(secs[i])
                    if not (1 <= pri <= 3) or not (1 <= sec <= 3):
                        raise ValueError(
                            f"Row {i+1}: Priority and Security must be between 1 and 3. "
                            f"Got Priority={pri}, Security={sec}."
                        )
                    raw_tasks.append({
                        "Task":     names[i].strip(),
                        "CPU":      int(cpus[i]),
                        "Memory":   int(mems[i]),
                        "Priority": pri,
                        "Security": sec,
                    })

            if not raw_tasks:
                raise ValueError("No valid task rows found in your dataset.")

            # ── STEP 2: AI Prediction ───────────────────────────────────────
            X        = np.array(
                [[t["CPU"], t["Memory"], t["Priority"], t["Security"]] for t in raw_tasks],
                dtype=float
            )
            X_scaled = ai_scaler.transform(X)
            preds    = ai_model.predict(X_scaled, verbose=0)  # shape: (n, 3)

            # ── STEP 3: First-pass assignment to VM-0/VM-1/VM-2 ────────────
            # Bucket every task into the 3 base VMs first
            buckets = {"VM-0": [], "VM-1": [], "VM-2": []}

            for i, task in enumerate(raw_tasks):
                idx = int(np.argmax(preds[i])) % 3
                vm  = VM_LABELS[idx]
                task["_base_vm"] = vm   # remember original assignment
                buckets[vm].append(task)

            # ── STEP 4: PRE-SCALE — check if any base VM will be overloaded ─
            # Calculate what avg CPU would be per base VM
            pre_avg = {}
            for vm in VM_LABELS:
                tasks_in = buckets[vm]
                if tasks_in:
                    pre_avg[vm] = round(sum(t["CPU"] for t in tasks_in) / len(tasks_in), 2)
                else:
                    pre_avg[vm] = 0.0

            # Spawn overflow VMs for overloaded base VMs BEFORE final assignment
            # vm_children: maps base VM → its overflow child VM id
            vm_children = {}   # e.g. {"VM-2": "VM-3"}
            local_pool  = list(VM_LABELS)  # FRESH per request — prevents cross-request KeyError

            for vm in VM_LABELS:
                if pre_avg[vm] > SCALE_UP_THRESHOLD:
                    if len(local_pool) < MAX_VMS:
                        new_id = f"VM-{len(local_pool)}"
                        local_pool.append(new_id)
                        vm_children[vm] = new_id
                        scale_up.append(
                            f"{vm} overloaded ({pre_avg[vm]}% avg CPU) → "
                            f"{new_id} spawned and added to pool"
                        )
                    else:
                        scale_up.append(
                            f"{vm} overloaded ({pre_avg[vm]}% avg CPU) → "
                            f"pool at max capacity ({MAX_VMS} VMs), cannot spawn"
                        )

            # ── STEP 5: Redistribute overflow tasks to child VMs ────────────
            # Child VM inherits the SAME profile as its parent (priority,
            # security, memory). Tasks are filtered so that the child only
            # receives tasks that match the parent profile — ensuring the
            # spawned VM is truly compatible with the workload it handles.
            #
            # Profile rules (from VM_PROFILE):
            #   VM-0 child → priority==1, security==1, memory<=2048
            #   VM-1 child → priority<=2, security<=2, memory<=4096
            #   VM-2 child → priority==3 OR security==3 OR memory>4096
            #
            # Any tasks that don't match are kept on the parent.
            # If nothing matches the profile filter, fall back to 50/50 split.

            for parent_vm, child_vm in vm_children.items():
                parent_tasks = buckets[parent_vm]
                profile      = VM_PROFILE.get(parent_vm, VM_PROFILE["VM-0"])

                if parent_vm == "VM-0":
                    # VM-0 profile: low priority, low security, low memory
                    match    = [t for t in parent_tasks
                                if t["Priority"] == 1
                                and t["Security"] == 1
                                and t["Memory"] <= profile["memory_max"]]
                    no_match = [t for t in parent_tasks if t not in match]

                elif parent_vm == "VM-1":
                    # VM-1 profile: mid priority (<=2), mid security (<=2), mid memory (<=4096)
                    match    = [t for t in parent_tasks
                                if t["Priority"] <= 2
                                and t["Security"] <= 2
                                and t["Memory"] <= profile["memory_max"]]
                    no_match = [t for t in parent_tasks if t not in match]

                else:  # VM-2
                    # VM-2 profile: high priority (3) OR high security (3) OR large memory
                    match    = [t for t in parent_tasks
                                if t["Priority"] == 3
                                or t["Security"] == 3
                                or t["Memory"] > 4096]
                    no_match = [t for t in parent_tasks if t not in match]

                # If profile filter gives us tasks → move matched tasks to child
                # If nothing matched → fall back to simple 50/50 split
                if match:
                    # Keep non-matching tasks on parent, move matched to child
                    # But also keep at least half on parent to avoid emptying it
                    split              = max(1, len(match) // 2)
                    buckets[parent_vm] = no_match + match[:split]
                    buckets[child_vm]  = match[split:]
                elif len(parent_tasks) > 1:
                    # Fallback: 50/50 split when no profile match found
                    split              = len(parent_tasks) // 2
                    buckets[parent_vm] = parent_tasks[:split]
                    buckets[child_vm]  = parent_tasks[split:]

            # ── STEP 6: Final assignment — flatten buckets into results ──────
            counts  = {vm: 0  for vm in local_pool}
            cpu_sum = {vm: 0  for vm in local_pool}
            t_names = {vm: [] for vm in local_pool}

            for vm_id, tasks_in in buckets.items():
                for task in tasks_in:
                    task["VM"]     = vm_id
                    task["Reason"] = get_reason(vm_id, task)
                    task["Risk"]   = task["Priority"] * task["Security"]
                    task.pop("_base_vm", None)

                    results.append(task)
                    counts[vm_id]  += 1
                    cpu_sum[vm_id] += task["CPU"]
                    t_names[vm_id].append(task["Task"])

            # ── STEP 7: Average CPU per VM (post-redistribution) ────────────
            avg_cpu = {}
            for vm_id in local_pool:
                n = counts[vm_id]
                avg_cpu[vm_id] = round(cpu_sum[vm_id] / n, 2) if n > 0 else 0.0

            # ── STEP 8: Scale DOWN — only decommission if child VM is idle ──
            # A spawned VM (VM-3/4/5) should only be removed if it truly has
            # no tasks AND its CPU is below threshold after redistribution.
            # Core VMs (VM-0/1/2) are NEVER decommissioned.
            for vm_id in list(local_pool):
                if vm_id in VM_LABELS:
                    continue   # never remove core VMs
                if counts.get(vm_id, 0) == 0 and avg_cpu.get(vm_id, 0) < SCALE_DOWN_THRESHOLD:
                    if len(local_pool) > MIN_VMS:
                        local_pool.remove(vm_id)
                        scale_down.append(
                            f"{vm_id} idle (0 tasks, {avg_cpu[vm_id]}% CPU) → "
                            f"decommissioned and removed from pool"
                        )
                        avg_cpu.pop(vm_id, None)
                        counts.pop(vm_id, None)
                        t_names.pop(vm_id, None)

            # ── STEP 9: VM Summary ──────────────────────────────────────────
            for vm_id in local_pool:
                parent   = get_parent(vm_id)
                profile  = VM_PROFILE.get(parent, VM_PROFILE["VM-0"])
                sample   = ", ".join(t_names.get(vm_id, [])[:3])
                is_child = vm_id not in VM_LABELS

                if is_child:
                    classification = (
                        f"Overflow VM — inherits {parent} profile "
                        f"(Priority ≤{profile['priority']}, "
                        f"Security ≤{profile['security']}, "
                        f"Memory ≤{profile['memory_max']}MB)"
                    )
                else:
                    classification = "Grouped based on AI workload classification"

                vm_summary.append({
                    "vm":     vm_id,
                    "count":  counts.get(vm_id, 0),
                    "sample": sample if sample else "—",
                    "reason": classification,
                })

            # ── STEP 10: VM Health ──────────────────────────────────────────
            for vm_id in local_pool:
                base = avg_cpu.get(vm_id, 0)
                cpu  = max(0, min(100, base + random.randint(-5, 5)))

                if cpu < 60:
                    status, color, state = "Healthy",  "green",  "Optimal CPU usage"
                elif cpu < 80:
                    status, color, state = "Warning",  "orange", "Moderate workload"
                else:
                    status, color, state = "Critical", "red",    "High CPU pressure"

                vm_health[vm_id] = {
                    "cpu":    cpu,
                    "status": status,
                    "color":  color,
                    "state":  state,
                }

            # ── Final template values ───────────────────────────────────────
            vm_count  = counts
            vm_load   = avg_cpu
            pool_size = len(local_pool)

        except Exception as e:
            traceback.print_exc()
            print(f"\n>>> SITAS ERROR: {e}\n")
            error_msg  = str(e)
            results    = []
            vm_count   = {"VM-0": 0, "VM-1": 0, "VM-2": 0}
            vm_load    = {"VM-0": 0, "VM-1": 0, "VM-2": 0}
            vm_summary = []
            vm_health  = {}
            scale_up   = []
            scale_down = []
            pool_size  = 3

    return render_template(
        "index.html",
        results    = results[:10],
        vm_count   = vm_count,
        vm_load    = vm_load,
        vm_summary = vm_summary,
        vm_health  = vm_health,
        scale_up   = scale_up,
        scale_down = scale_down,
        pool_size  = pool_size,
        error_msg  = error_msg,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
