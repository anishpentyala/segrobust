"""Reproducible synchronization witnesses, not probabilistic thread races."""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from segrobust import corrupt
from segrobust.api import CORRUPTIONS


def test_seeded_library_calls_do_not_cross_contaminate(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    original = CORRUPTIONS['gaussian_noise']['fn']

    def slow_first(image, severity):
        if threading.current_thread().name.endswith('_0'):
            entered.set()
            assert release.wait(5), 'test synchronization timeout'
        else:
            second_entered.set()
            assert release_second.wait(5), 'second test synchronization timeout'
        return original(image, severity)

    image = np.arange(16*16*3, dtype=np.uint8).reshape(16,16,3)
    want_first = corrupt(image, 'gaussian_noise', 3, seed=7)
    want_second = corrupt(image, 'gaussian_noise', 3, seed=11)
    monkeypatch.setitem(CORRUPTIONS['gaussian_noise'], 'fn', slow_first)
    np.random.seed(137)
    before = np.random.get_state()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='rng-control') as pool:
        first = pool.submit(corrupt, image, 'gaussian_noise', 3, 7)
        assert entered.wait(5), 'first library call did not enter'

        def second_call():
            second_started.set()
            return corrupt(image, 'gaussian_noise', 3, seed=11)

        second = pool.submit(second_call)
        assert second_started.wait(5), 'second thread did not start'
        # On old code the second corruption enters while the first is
        # paused. Force non-LIFO completion to expose state cross-talk.
        # A library lock keeps it outside until the first finishes.
        second_entered.wait(.1)
        release.set()
        try:
            actual_first = first.result(timeout=5)
        finally:
            release_second.set()
        actual_second = second.result(timeout=5)
    after = np.random.get_state()
    assert np.array_equal(actual_first, want_first), 'concurrent seeded library call changed first output'
    assert np.array_equal(actual_second, want_second), 'concurrent seeded library call changed second output'
    assert before[0] == after[0] and np.array_equal(before[1], after[1]), 'concurrent library calls changed caller RNG state'
    assert before[2:] == after[2:]


def test_nested_seeded_context_restores_outer_draws():
    from segrobust.api import _seeded
    np.random.seed(13)
    before = np.random.get_state()
    expected = np.random.RandomState(3).rand(3)
    with _seeded(3):
        first = np.random.rand()
        with _seeded(7):
            np.random.rand(10)
        rest = np.random.rand(2)
    actual = np.concatenate(([first], rest))
    np.testing.assert_array_equal(actual, expected)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_corruption_exception_releases_lock_and_restores_rng(monkeypatch):
    image = np.zeros((16,16,3), dtype=np.uint8)
    original = CORRUPTIONS['gaussian_noise']['fn']

    def fail(image, severity):
        np.random.rand(3)
        raise RuntimeError('synthetic corruption failure')

    np.random.seed(171)
    before = np.random.get_state()
    monkeypatch.setitem(CORRUPTIONS['gaussian_noise'], 'fn', fail)
    import pytest
    with pytest.raises(RuntimeError, match='synthetic'):
        corrupt(image, 'gaussian_noise', 3, seed=7)
    monkeypatch.setitem(CORRUPTIONS['gaussian_noise'], 'fn', original)
    with ThreadPoolExecutor(max_workers=1) as pool:
        actual = pool.submit(corrupt, image, 'gaussian_noise', 3, 7).result(timeout=5)
    assert actual.shape == image.shape
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_unseeded_context_waits_for_seeded_owner():
    from segrobust.api import _seeded
    owner_entered, release = threading.Event(), threading.Event()
    visitor_started, visitor_entered = threading.Event(), threading.Event()

    def owner():
        with _seeded(7):
            owner_entered.set()
            assert release.wait(5)
            np.random.rand(3)

    def visitor():
        visitor_started.set()
        with _seeded(None):
            visitor_entered.set()
            return np.random.rand(3)

    np.random.seed(171)
    expected = np.random.RandomState(171).rand(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(owner)
        assert owner_entered.wait(5)
        b = pool.submit(visitor)
        assert visitor_started.wait(5)
        overlapped = visitor_entered.wait(.1)
        release.set()
        a.result(timeout=5)
        actual = b.result(timeout=5)
    assert not overlapped, 'unseeded library context must not enter a seeded owner'
    np.testing.assert_array_equal(actual, expected)


def test_fork_child_does_not_inherit_lock_owned_by_vanished_thread(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import pytest
    import segrobust
    if not hasattr(os, 'fork'):
        pytest.skip('fork-specific behavior')
    script = tmp_path / 'fork_check.py'
    script.write_text('''import os,threading,signal
from segrobust.api import _seeded
entered=threading.Event();release=threading.Event()
def owner():
 with _seeded(3):
  entered.set();release.wait(10)
t=threading.Thread(target=owner);t.start();assert entered.wait(2)
pid=os.fork()
if pid==0:
 signal.alarm(2)
 with _seeded(4):pass
 os._exit(0)
_,status=os.waitpid(pid,0);release.set();t.join()
assert os.waitstatus_to_exitcode(status)==0, 'fork child inherited a held library lock'
''')
    env = dict(os.environ, PYTHONPATH=str(Path(segrobust.__file__).resolve().parent.parent))
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          text=True, capture_output=True, timeout=15)
    assert done.returncode == 0, done.stderr


def test_fork_inside_nested_seeded_owner_reenters_in_child(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import pytest
    import segrobust
    if not hasattr(os, 'fork'):
        pytest.skip('fork-specific behavior')
    script = tmp_path / 'nested_fork.py'
    script.write_text('''import os,signal
import numpy as np
from segrobust.api import _seeded
np.random.seed(19)
original=np.random.get_state()
with _seeded(3):
 with _seeded(7):
  pid=os.fork()
  if pid==0:
   signal.alarm(3)
   with _seeded(11):
    np.random.rand(2)
   with _seeded(None):
    np.random.rand(2)
   # Also exercise the surrounding inherited context exits.
  else:
   _,status=os.waitpid(pid,0)
   assert os.waitstatus_to_exitcode(status)==0
if pid==0:
 with _seeded(13):np.random.rand(2)
 os._exit(0)
after=np.random.get_state()
assert np.array_equal(original[1],after[1]) and original[2:]==after[2:]
''')
    env = dict(os.environ, PYTHONPATH=str(Path(segrobust.__file__).resolve().parent.parent))
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          text=True, capture_output=True, timeout=15)
    assert done.returncode == 0, done.stderr


def test_seed_rejection_does_not_leave_lock_held():
    from segrobust.api import _seeded
    import pytest
    with pytest.raises(ValueError):
        with _seeded(-1):
            pass
    with ThreadPoolExecutor(max_workers=1) as pool:
        def enter():
            with _seeded(7):
                return np.random.rand()
        assert pool.submit(enter).result(timeout=5) == np.random.RandomState(7).rand()
