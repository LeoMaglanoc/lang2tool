"""Helpers for bounded simulator shutdown in subprocess evaluation runs."""


# Close Isaac Lab app with a timeout to avoid subprocess hangs in headless e2e runs.
def close_simulation_app_with_timeout(
    simulation_app,
    *,
    timeout_sec: float = 15.0,
    log_warn_fn=None,
    force_exit_fn=None,
) -> bool:
    """Close simulation app and return True; force-exit and return False on timeout."""
    import os
    import threading

    close_done = {"ok": False}

    def _close_simulation_app():
        simulation_app.close()
        close_done["ok"] = True

    close_thread = threading.Thread(target=_close_simulation_app, daemon=True)
    close_thread.start()
    close_thread.join(timeout=timeout_sec)
    if close_done["ok"]:
        return True

    if log_warn_fn is not None:
        log_warn_fn("simulation_app.close() timed out; forcing process exit.")
    if force_exit_fn is None:
        force_exit_fn = os._exit
    force_exit_fn(0)
    return False
