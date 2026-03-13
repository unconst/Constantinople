"""Pytest configuration — starts mock miners for integration tests."""

import multiprocessing
import time

import pytest
import requests


def _start_mock_miner(port: int):
    """Start a mock miner in a subprocess."""
    import uvicorn
    from multi_gpu_miner import create_miner_app
    app, _ = create_miner_app(num_workers=1)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait_for_server(url: str, timeout: float = 10) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


@pytest.fixture(scope="session", autouse=True)
def mock_miners():
    """Start mock miners on ports 19191 and 19192 for integration tests."""
    procs = []
    for port in [19191, 19192]:
        p = multiprocessing.Process(target=_start_mock_miner, args=(port,), daemon=True)
        p.start()
        procs.append(p)

    for port in [19191, 19192]:
        if not _wait_for_server(f"http://127.0.0.1:{port}/health"):
            for p in procs:
                p.terminate()
            pytest.fail(f"Mock miner on port {port} failed to start")

    yield

    for p in procs:
        p.terminate()
        p.join(timeout=5)
