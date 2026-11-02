"""Run arms of coupled_arm.py on this step's GPU(s), SLOTS processes per GPU at a time, longest first.

python coupled_queue.py OUT ARMS.json SHARD N_SHARDS SLOTS
ARMS.json: [{"name", "seed", "stop_after", "cost", "manifest", "relax": [...], "driver": [...], optional "data": [...]}].
Suite mode only (manifest required). Default data flags are round 1's grid: --shift-step 1 --shift-range 2 (+-2 px in
1 px steps, never re-centred); an arm's "data" list replaces them (the reproduction run uses the old suite grid).
Arms whose output already has DONE, or that another step claimed (OUT/<name>.claim, or .$CLAIM_SUFFIX), are skipped. Environment from env.sh (PY, SRC).
"""
import json, os, queue, subprocess, sys, threading, time
from pathlib import Path

R = Path('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102')  # round 14: runs round 14's driver copy
out, arms_file, shard, n_shards, slots = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
arms = sorted(json.loads(arms_file.read_text()), key=lambda a: (-a["cost"], a["name"]))[shard::n_shards]
out.mkdir(parents=True, exist_ok=True)
work = queue.Queue()
for arm in arms:
    if not (out / arm["name"] / "DONE").exists():
        work.put(arm)
for f in out.glob("*.claim"):  # round 17: one queue at a time; a claim of an arm without DONE is from a dead job
    if not (out / f.stem / "DONE").exists():
        f.unlink()
status, lock = {}, threading.Lock()
PY, SRC = os.environ["PY"], os.environ["SRC"]
job = os.environ.get("SLURM_JOB_ID", "local") + "." + os.environ.get("SLURM_STEP_ID", "x")


def worker(gpu):
    while True:
        try:
            arm = work.get_nowait()
        except queue.Empty:
            return
        try:  # claim: several steps (pools) share one arms list; the first to create the claim runs the arm
            os.close(os.open(out / f"{arm['name']}.{os.environ.get('CLAIM_SUFFIX', 'claim')}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        data = ["--manifest", arm["manifest"], "--eval-every", "500", *arm.get("data", ["--shift-step", "1", "--shift-range", "2"])]
        resume = (out / arm["name"] / "checkpoint_0000.npz").exists()  # round 17: restart from the newest checkpoint
        cmd = [PY, str(R / "tools/coupled_arm.py"), "--source", SRC, *data,
               "--output", str(out / arm["name"]), "--seed", str(arm["seed"]), "--stop-after", str(arm["stop_after"]),
               *arm.get("driver", []), *(["--resume"] if resume else []), "--", *arm["relax"]]
        if (out / arm["name"]).exists() and not resume:  # partial earlier attempt: keep it aside, the driver refuses an existing output
            (out / arm["name"]).rename(out / f"{arm['name']}.partial_{int(time.time())}")
        t0 = time.monotonic()
        with open(out / f"{arm['name']}.log", "a" if resume else "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        with lock:
            status[arm["name"]] = {"rc": rc, "gpu": gpu, "wall_s": time.monotonic() - t0}
            (out / f"arms_status_{job}_{shard}.json").write_text(json.dumps(status, indent=1) + "\n")
            print(arm["name"], f"rc={rc}", f"gpu={gpu}", f"wall {time.monotonic() - t0:.0f} s", time.strftime("%T"), flush=True)


gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
threads = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(slots)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ALL_DONE", flush=True)
sys.exit(1 if any(v["rc"] for v in status.values()) else 0)  # a failed arm fails the step
