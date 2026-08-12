"""One entry point for the whole pipeline.

    python -m volradar.cli fetch      # download raw series (network)
    python -m volradar.cli build      # assemble feature panels
    python -m volradar.cli evaluate   # accuracy: fixed split, pooled OOS, walk-forward
    python -m volradar.cli ablation   # what each factor family is worth
    python -m volradar.cli methods    # methodology bake-off: which modelling changes help
    python -m volradar.cli events     # did it flag real volatility spikes?
    python -m volradar.cli decision   # would acting on it have helped?
    python -m volradar.cli simulate   # validate the method on known-truth synthetic markets
    python -m volradar.cli train      # fit production models, write the snapshot
    python -m volradar.cli all        # everything above, in order
    python -m volradar.cli serve      # run the API + dashboard

Each stage is idempotent and caches its output, so re-running `all` after a code change
only redoes what is downstream of it. Add `--force` to rebuild from scratch.
"""

from __future__ import annotations

import argparse
import sys
import time

from volradar import config


def _stage(name: str, fn, *args, **kwargs):
    print(f"\n{'=' * 70}\n>> {name}\n{'=' * 70}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"-- {name} finished in {time.time() - t0:.1f}s")
    return result


def cmd_fetch(args):
    from volradar.data.fetch import fetch_all
    fetch_all(force=args.force)


def cmd_build(args):
    from volradar.features.assemble import build_panels
    build_panels(force=args.force)


def cmd_evaluate(args):
    from volradar.backtest.evaluate import main as run
    run()


def cmd_ablation(args):
    from volradar.backtest.ablation import main as run
    run()


def cmd_methods(args):
    from volradar.backtest.methods import main as run
    run()


def cmd_select(args):
    from volradar.backtest.select_features import main as run
    run()


def cmd_events(args):
    from volradar.backtest.events import main as run
    run()


def cmd_decision(args):
    from volradar.backtest.decision import main as run
    run()


def cmd_simulate(args):
    from volradar.simulate.validate import main as validate
    from volradar.simulate.scenarios import main as scenarios
    validate()
    scenarios()


def cmd_train(args):
    from volradar.models.train import main as run
    run()


def cmd_oos(args):
    from volradar.backtest.oos import load_or_build
    for panel in ("market", "sector"):
        load_or_build(panel, force=args.force)


def cmd_all(args):
    _stage("fetch", cmd_fetch, args)
    _stage("build panels", cmd_build, args)
    _stage("out-of-sample predictions", cmd_oos, args)
    _stage("evaluate", cmd_evaluate, args)
    _stage("ablation", cmd_ablation, args)
    _stage("methodology bake-off", cmd_methods, args)
    _stage("event study", cmd_events, args)
    _stage("decision simulation", cmd_decision, args)
    _stage("known-truth simulation", cmd_simulate, args)
    _stage("train production models", cmd_train, args)
    print(f"\nAll artifacts in {config.REPORT_DIR}")


def cmd_serve(args):
    import uvicorn
    print(f"API   -> http://{args.host}:{args.port}/api/health")
    if (config.ROOT / "frontend" / "dist").exists():
        print(f"Board -> http://{args.host}:{args.port}/")
    else:
        print("Dashboard not built yet: cd frontend && npm install && npm run build")
    uvicorn.run("volradar.api.server:app", host=args.host, port=args.port, reload=args.reload)


COMMANDS = {
    "fetch": cmd_fetch, "build": cmd_build, "oos": cmd_oos, "evaluate": cmd_evaluate,
    "ablation": cmd_ablation, "methods": cmd_methods, "select": cmd_select,
    "events": cmd_events, "decision": cmd_decision, "simulate": cmd_simulate,
    "train": cmd_train, "all": cmd_all, "serve": cmd_serve,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="volradar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("--force", action="store_true",
                   help="ignore caches and rebuild this stage from scratch")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload the API on edits")
    args = p.parse_args(argv)

    config.ensure_dirs()
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
