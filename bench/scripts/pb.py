"""Shared runtime for the public benchmark runners (Linux/macOS; every path comes from the environment).

Every recall call gets a scratch HOME/USERPROFILE (store.rs resolves ~/.recall from it), so a run never
opens anyone's real recall store.

  RECALL_BIN          recall binary under test (required)
  RECALL_MODEL_DIR    dir that holds models--Xenova--bge-small-en-v1.5 (copied into each scratch ~/.recall/models);
                      only needed when a bge channel runs
  ORT_DYLIB_PATH      libonnxruntime.so (needed for the bge channel)
  BENCH_WORK          scratch + output dir (default ./bench-work)
  BENCH_DATA          downloaded datasets (default ./bench-data, filled by fetch_data.py)
  BENCH_CPUS          cpu list for taskset, e.g. 0-7 or 0,2,4 (default: unpinned). Each worker gets one cpu.
  BENCH_LOAD_MAX      pause new work while the 1-minute load average is above this (default 18)
  BENCH_MEM_FLOOR_GB  pause while MemAvailable is below this (default 8)

Launch long runs under `nice -n 19 ionice -c3` so they yield to everything else on the box.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

WORK = os.path.abspath(os.environ.get("BENCH_WORK", "bench-work"))
DATA = os.path.abspath(os.environ.get("BENCH_DATA", "bench-data"))
LOAD_MAX = float(os.environ.get("BENCH_LOAD_MAX", "18"))
MEM_FLOOR_GB = float(os.environ.get("BENCH_MEM_FLOOR_GB", "8"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_ENVH = {}


def env_hashes():
    """sha256 of the ONNX model and the ORT library in use (cached): same executable + different model = different experiment."""
    if not _ENVH:
        ort = os.environ.get("ORT_DYLIB_PATH")
        _ENVH["ort_sha256"] = sha256(ort) if ort and os.path.isfile(ort) else None
        mdl = None
        for root, _, fs in os.walk(os.environ.get("RECALL_MODEL_DIR", "")):
            for f in fs:
                if f == "model.onnx":
                    mdl = os.path.join(root, f)
        _ENVH["model_onnx_sha256"] = sha256(mdl) if mdl else None
    return dict(_ENVH)


def pin_identity(path, ident):
    """First call writes the run identity; later calls must match it, so a resume can never mix two builds/settings."""
    ident = dict(ident, exe_sha256=exe_sha(), **env_hashes())
    if os.path.exists(path):
        old = json.load(open(path))
        if old != ident:
            raise SystemExit(f"{path} was made with different settings: was {old}, now {ident}; use a fresh BENCH_WORK/--out")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(ident, open(path, "w"), indent=1)


def exe():
    p = os.environ.get("RECALL_BIN")
    if not p or not os.path.isfile(p):
        raise SystemExit("set RECALL_BIN to the recall binary under test")
    return os.path.abspath(p)


def exe_sha():
    return sha256(exe())


def cpu_list():
    spec = os.environ.get("BENCH_CPUS", "").strip()
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def cpu_for(i):
    c = cpu_list()
    return str(c[i % len(c)]) if c else None


def cpu_all():
    c = cpu_list()
    return ",".join(map(str, c)) if c else None


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def mem_avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) / 1048576
    return 1e9


def wait_heavy(log=print, poll_s=30, max_wait_s=2 * 3600):
    """Block while the box is busy (load or memory); abort the run if it never clears."""
    t0 = time.time()
    n = 0
    while True:
        l, m = load1(), mem_avail_gb()
        if l <= LOAD_MAX and m >= MEM_FLOOR_GB:
            return time.time() - t0
        if n % 10 == 0:
            log(f"guard: load1={l:.1f} mem_avail={m:.1f}GB - paused")
        n += 1
        if time.time() - t0 > max_wait_s:
            raise SystemExit("guard never cleared")
        time.sleep(poll_s)


def make_home(home, models=True):
    os.makedirs(os.path.join(home, ".recall"), exist_ok=True)
    os.makedirs(os.path.join(home, "tmp"), exist_ok=True)
    dst = os.path.join(home, ".recall", "models")
    src = os.environ.get("RECALL_MODEL_DIR")
    if models and src and not os.path.isdir(dst):
        shutil.copytree(src, dst)
    return home


def fresh_db(home):
    r = os.path.join(home, ".recall")
    for n in ("recall.db", "recall.db-wal", "recall.db-shm", "daemon.json"):
        p = os.path.join(r, n)
        if os.path.exists(p):
            os.remove(p)


SCRUB_STORE_REDACTED = "0"
print(f"bench: RECALL_SCRUB_STORE_REDACTED={SCRUB_STORE_REDACTED} (benchmark stores keep unredacted text)", file=sys.stderr, flush=True)


def env_for(home, vault, abstain):
    e = {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "TMPDIR": os.path.join(home, "tmp"), "USERPROFILE": home, "HOME": home,
        "HF_HOME": os.path.join(home, "hf"), "XDG_CACHE_HOME": os.path.join(home, "cache"),
        "RECALL_VAULT": vault, "RECALL_DENOISE_MODE": "off", "RECALL_TIERS": "all",
        "RECALL_BACKEND": "cpu", "RECALL_ABSTAIN": "on" if abstain else "off",
        # Benchmark stores index the ORIGINAL text: since main d1aef48 (PR #16) a new binary may store redacted
        # text by default, which would score a different corpus than the published benchmark.
        "RECALL_SCRUB_STORE_REDACTED": SCRUB_STORE_REDACTED,
        # recall's embed runs ORT threads = cpus it may use PLUS a rayon pool of the same size; cap the pool so a run
        # pinned to N cpus holds about N busy threads instead of 2N (measured: 16 threads on 8 cpus -> 9)
        "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "1"),
    }
    if os.environ.get("ORT_DYLIB_PATH"):
        e["ORT_DYLIB_PATH"] = os.environ["ORT_DYLIB_PATH"]
    return e


def _cmd(args, cpus):
    return (["taskset", "-c", cpus] if cpus else []) + [exe()] + args


def run(args, env, timeout=None, cpus=None):  # no default timeout: index/embed time scales with the corpus (FiQA embed > 2 h)
    t = time.time()
    r = subprocess.run(_cmd(args, cpus), env=env, cwd=env["USERPROFILE"], capture_output=True, timeout=timeout)
    return {"rc": r.returncode, "out": r.stdout.decode("utf-8", "replace"),
            "err": r.stderr.decode("utf-8", "replace"), "ms": (time.time() - t) * 1000}


def build_store(home, vault, cpus=None, embed=True, log=print):
    """index (+ embed) a vault into a scratch home. Returns rc/ms/tails for the build record."""
    make_home(home, models=embed)
    env = env_for(home, vault, False)
    res = {}
    r = run(["index"], env, cpus=cpus)
    res["index"] = {"rc": r["rc"], "ms": r["ms"], "tail": (r["out"] + r["err"])[-600:]}
    if r["rc"]:
        raise RuntimeError("index failed: " + r["err"][-600:])
    if embed:
        r = run(["embed", "--batch", "64", "--allow-benchmark-regression"], env, cpus=cpus)
        res["embed"] = {"rc": r["rc"], "ms": r["ms"], "tail": (r["out"] + r["err"])[-600:]}
        if r["rc"]:
            raise RuntimeError("embed failed: " + r["err"][-600:])
    r = run(["status"], env, cpus=cpus)
    res["status"] = (r["out"] + r["err"])[-800:]
    return res


class Daemon:
    """One scratch daemon per (home, abstain) pass; started and stopped by the process that owns it."""

    def __init__(self, home, vault, abstain, logdir, cpus=None):
        os.makedirs(logdir, exist_ok=True)
        self.env = env_for(home, vault, abstain)
        self.dj = os.path.join(home, ".recall", "daemon.json")
        if os.path.exists(self.dj):
            os.remove(self.dj)
        tag = "%s-%d" % ("abs" if abstain else "noabs", int(time.time()))
        self.fo = open(os.path.join(logdir, f"daemon-{tag}.out"), "wb")
        self.fe = open(os.path.join(logdir, f"daemon-{tag}.err"), "wb")
        self.p = subprocess.Popen(_cmd(["daemon", "up"], cpus), env=self.env, cwd=home, stdout=self.fo, stderr=self.fe)
        for _ in range(600):
            if os.path.exists(self.dj) or self.p.poll() is not None:
                break
            time.sleep(1)
        time.sleep(1)
        self.info = json.load(open(self.dj)) if os.path.exists(self.dj) else {}
        if self.p.poll() is not None:
            self.stop()
            raise RuntimeError(f"daemon failed to start: {self.info}")

    def query(self, text, channels, top):
        return run(["query", "--via-daemon", "--json", "--top", str(top), "--channels", channels, "--", text],
                   self.env, timeout=300)

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.fo.close()
        self.fe.close()
        if os.path.exists(self.dj):
            os.remove(self.dj)


def parse_hits(r):
    """('ok', hits) or ('error', []). An empty hit list means the build abstained or found nothing."""
    if r["rc"] != 0:
        return "error", []
    try:
        h = json.loads(r["out"]) if r["out"].strip() else None
    except json.JSONDecodeError:
        return "error", []
    if not isinstance(h, list):
        return "error", []
    return "ok", h


def stem_of(hit):
    return os.path.basename(hit["vault_path"].replace("\\", "/"))[:-3]
