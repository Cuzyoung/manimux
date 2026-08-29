#!/usr/bin/env python3
"""Launch TensorBoard on loopback despite the cluster image's bind-all default."""

from __future__ import annotations

import argparse
import signal

from tensorboard import program


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--logdir")
    source.add_argument("--logdir-spec")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--reload-interval", type=float, default=10.0)
    args = parser.parse_args()

    tensorboard = program.TensorBoard()
    argv = [None]
    if args.logdir is not None:
        argv.extend(["--logdir", args.logdir])
    else:
        argv.extend(["--logdir_spec", args.logdir_spec])
    argv.extend(
        ["--port", str(args.port), "--reload_interval", str(args.reload_interval)]
    )
    tensorboard.configure(argv=argv)
    tensorboard.flags.bind_all = False
    tensorboard.flags.host = "127.0.0.1"
    print(tensorboard.launch(), flush=True)
    signal.pause()


if __name__ == "__main__":
    main()
