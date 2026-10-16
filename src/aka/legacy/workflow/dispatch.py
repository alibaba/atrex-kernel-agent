"""Run framework campaigns in child processes with inherited Workflow dependencies."""
from __future__ import annotations
import os, signal, sys, time
import traceback
from aka.legacy.agent.process import descendant_process_commands
from aka.legacy.execution.recovery import ENVIRONMENT_TEMPFAIL, EnvironmentUnavailable, environment_is_blocked


class _CampaignProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                return
            time.sleep(.05)


def dispatch_framework_campaigns(frameworks, run_framework) -> int:
    """Preserve per-framework budgets, independent failures and group cancellation.

    The synchronous POSIX startup forks before any campaign starts worker threads.
    Each child borrows the assembled module selections and loaded Problem; it does
    not enter Bootstrap, reload providers, or dispose the parent's composition.
    """
    children = []
    failed = []

    def stop_children() -> None:
        # Framework children and every coding-agent session deliberately use
        # separate process groups.  Capture the whole descendant group set
        # before signaling: once a framework coordinator exits, its Claude
        # session is reparented to PID 1 and can no longer be discovered from
        # the dispatcher, which previously left orphan optimizers behind.
        process_groups: set[int] = set()
        for _, _, proc in children:
            if proc.poll() is not None:
                continue
            try:
                process_groups.add(os.getpgid(proc.pid))
            except ProcessLookupError:
                continue
            for pid, _argv in descendant_process_commands(proc.pid):
                try:
                    process_groups.add(os.getpgid(pid))
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                proc.wait(timeout=10)

        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                proc.wait(timeout=3)

        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                proc.wait(timeout=5)


    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }

    def interrupt_dispatch(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for handled_signal in handled_signals:
        signal.signal(handled_signal, interrupt_dispatch)
    try:
        for framework in frameworks:
            sys.stdout.flush()
            sys.stderr.flush()
            ready_read, ready_write = os.pipe()
            # Register a child only after setsid is acknowledged. Defer parent
            # cancellation until then, so it cannot orphan an unregistered
            # child or signal the parent's own process group.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, (*handled_signals, signal.SIGINT))
            try:
                pid = os.fork()
            except BaseException:
                os.close(ready_read)
                os.close(ready_write)
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                raise
            if pid == 0:
                os.close(ready_read)
                result = 1
                try:
                    os.setsid()
                    os.write(ready_write, b"1")
                    os.close(ready_write)
                    for sig, handler in previous_handlers.items():
                        signal.signal(sig, handler)
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    os.environ["ATREX_ENVIRONMENT_RECOVERY_OWNER"] = "0"
                    result = run_framework(framework)
                except KeyboardInterrupt:
                    result = 130
                except EnvironmentUnavailable:
                    result = ENVIRONMENT_TEMPFAIL
                except SystemExit as error:
                    result = error.code if type(error.code) is int else (0 if error.code is None else 1)
                except BaseException:
                    traceback.print_exc()
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(result if type(result) is int and 0 <= result <= 255 else 1)
            os.close(ready_write)
            proc = _CampaignProcess(pid)
            try:
                # Cancellation must never mistake the parent's group for the
                # child's group before setsid has completed.
                if os.read(ready_read, 1) != b"1":
                    proc.wait(timeout=5)
                    raise RuntimeError("framework child failed to create its process group")
            finally:
                os.close(ready_read)
                children.append((framework, "", proc))
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            print(f"[orchestrator] dispatched framework={framework} pid={pid}", flush=True)

        remaining = {proc.pid: (framework, proc) for framework, _, proc in children}
        while remaining:
            for pid, (framework, proc) in list(remaining.items()):
                returncode = proc.poll()
                if returncode is None:
                    continue
                remaining.pop(pid)
                print(
                    f"[orchestrator] framework={framework} finished exit={returncode}",
                    flush=True,
                )
                if returncode != 0:
                    failed.append((framework, returncode))
                if returncode == ENVIRONMENT_TEMPFAIL or environment_is_blocked():
                    stop_children()
                    return ENVIRONMENT_TEMPFAIL
            if remaining:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print(
            "[orchestrator] interrupt: stopping framework campaigns",
            file=sys.stderr,
            flush=True,
        )
        stop_children()
        return 130
    except BaseException:
        stop_children()
        raise
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)

    if failed:
        summary = ", ".join(f"{name}={code}" for name, code in failed)
        print(
            f"[orchestrator] framework campaign failures: {summary}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0
