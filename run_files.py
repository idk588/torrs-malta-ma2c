"""Crash-safe run files and per-episode compute cost (spec section 7).

Every per-episode file in a run store (Malta/runs/<name>/) is written to <file>.tmp, flushed
to disk and then os.replace'd onto <file>, so a crash or a kill mid-write never leaves a
half-written CSV. A PermissionError (for example the file open in Excel) is retried
`retries` times, `wait_s` apart (6 x 10 s by default), and then raised.

episode_cost.csv gets one row per training episode (COST_COLUMNS) for the RQ2 compute-cost
metric. MemorySampler reads this python process's private bytes and working set through
GetProcessMemoryInfo (ctypes; psutil is not installed and would read the same two
counters). SUMO runs in its own process and is not included.

Units: GB = 2**30 bytes and MB = 2**20 bytes, as Task Manager and Explorer show them.
"""
import ctypes
import logging
import math
import numbers
import os
import sys
import threading
import time

import numpy as np

RETRIES = 6
WAIT_S = 10.0

COST_COLUMNS = ['episode', 'step', 'seed', 'wall_s', 'rollout_s', 'update_s', 'save_s',
                'peak_private_gb', 'peak_working_set_gb', 'ckpt_data_mb', 'ckpt_meta_mb', 'lost']

GB = float(2 ** 30)
MB = float(2 ** 20)


# --------------------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------------------
def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _flush_to_disk(path):
    """Make the written bytes durable before the rename, so a power cut cannot leave the
    renamed file empty. FlushFileBuffers flushes the file whichever handle wrote it."""
    with open(path, 'rb+') as f:
        os.fsync(f.fileno())


def _replace_atomically(write_tmp, path, retries, wait_s):
    """write_tmp(path + '.tmp'), then os.replace it onto path.

    The whole attempt (write and replace) is retried on PermissionError: `retries` more
    times, `wait_s` apart, then the error is raised. Any failure removes the .tmp file.
    """
    tmp = path + '.tmp'
    attempt = 0
    while True:
        try:
            write_tmp(tmp)
            _flush_to_disk(tmp)
            os.replace(tmp, path)
            return
        except PermissionError as e:
            if attempt >= retries:
                _remove_quietly(tmp)
                raise
            attempt += 1
            logging.warning('Write to %s blocked (%s); retry %d of %d in %g s'
                            % (path, e, attempt, retries, wait_s))
            time.sleep(wait_s)
        except BaseException:
            _remove_quietly(tmp)
            raise


def atomic_write_text(path, text, retries=RETRIES, wait_s=WAIT_S):
    """Write text (with '\\n' line ends) to path atomically: UTF-8, text mode, so the
    platform line end (CRLF on Windows), as pandas and np.savetxt write the other CSVs."""
    def write(tmp):
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text)
    _replace_atomically(write, path, retries, wait_s)


def atomic_write_csv(df, path, retries=RETRIES, wait_s=WAIT_S, **to_csv_kwargs):
    """df.to_csv(path, **to_csv_kwargs), atomically. The bytes are those of a plain to_csv."""
    _replace_atomically(lambda tmp: df.to_csv(tmp, **to_csv_kwargs), path, retries, wait_s)


def atomic_savetxt(arr, path, retries=RETRIES, wait_s=WAIT_S, **kw):
    """np.savetxt(path, arr, **kw), atomically. The bytes are those of a plain savetxt."""
    _replace_atomically(lambda tmp: np.savetxt(tmp, arr, **kw), path, retries, wait_s)


# --------------------------------------------------------------------------------------
# episode_cost.csv
# --------------------------------------------------------------------------------------
def _format_cost_value(value):
    """Integers as integers, floats to 4 decimals, missing values as NaN (which both
    pandas and PowerShell's [double]::Parse read back)."""
    if value is None:
        return 'NaN'
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        value = float(value)
        return 'NaN' if math.isnan(value) else '%.4f' % value
    raise TypeError('episode_cost.csv takes numbers only, got %r' % (value,))


def append_cost_row(path, row, retries=RETRIES, wait_s=WAIT_S):
    """Append one row to episode_cost.csv by an atomic rewrite of the existing text plus
    one line. The header is written only when the file is new (or empty); on resume the
    earlier rows are kept as they are. `lost` defaults to 0; every other column is required.
    """
    unknown = sorted(set(row) - set(COST_COLUMNS))
    if unknown:
        raise ValueError('episode_cost.csv has no column(s) %s' % unknown)
    missing = [c for c in COST_COLUMNS if c != 'lost' and c not in row]
    if missing:
        raise ValueError('cost row lacks column(s) %s' % missing)
    values = dict(row)
    values.setdefault('lost', 0)
    line = ','.join(_format_cost_value(values[c]) for c in COST_COLUMNS)
    header = ','.join(COST_COLUMNS)

    def write(tmp):
        existing = ''
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig') as f:
                existing = f.read()
        if existing.strip():
            first = existing.splitlines()[0].strip()
            if first != header:
                raise ValueError('%s starts %r, not the episode_cost.csv header' % (path, first))
            if not existing.endswith('\n'):
                existing += '\n'
            text = existing + line + '\n'
        else:
            text = header + '\n' + line + '\n'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text)
    _replace_atomically(write, path, retries, wait_s)


def file_size_mb(path):
    """Size of path in MB, or NaN when it does not exist."""
    try:
        return os.path.getsize(path) / MB
    except OSError:
        return math.nan


# --------------------------------------------------------------------------------------
# Process memory
# --------------------------------------------------------------------------------------
_get_memory_info = None

if sys.platform == 'win32':
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]

    # Private WinDLL instances, so these prototypes do not change ctypes.windll for others.
    # Without restype the pseudo-handle (-1) comes back as a truncated int and the call
    # fails, leaving the counters at zero.
    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    try:
        _get_memory_info = _kernel32.K32GetProcessMemoryInfo
    except AttributeError:  # before Windows 7 the function lives only in psapi.dll
        _get_memory_info = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    _get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                                 wintypes.DWORD]
    _get_memory_info.restype = wintypes.BOOL


def process_memory_bytes():
    """(private bytes, working set bytes) of this process. Raises OSError on failure."""
    if _get_memory_info is None:
        raise OSError('process memory is only read on Windows')
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    if not _get_memory_info(_kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PrivateUsage), int(counters.WorkingSetSize)


class MemorySampler(threading.Thread):
    """Daemon thread that samples this process's private bytes and working set every
    `interval_s` seconds and keeps the maxima since the last take_episode_peaks().

    Start it once before training; the Trainer calls take_episode_peaks() after each
    episode's checkpoint save.
    """

    def __init__(self, interval_s=1.0):
        super().__init__(name='MemorySampler', daemon=True)
        self.interval_s = interval_s
        self._peaks_lock = threading.Lock()
        self._halt = threading.Event()
        self._peak_private = None
        self._peak_working_set = None
        self._warned = False

    def run(self):
        while not self._halt.is_set():
            self._sample()
            self._halt.wait(self.interval_s)

    def stop(self):
        self._halt.set()

    def _sample(self):
        try:
            private, working_set = process_memory_bytes()
        except OSError as e:
            if not self._warned:
                self._warned = True
                logging.warning('MemorySampler: cannot read process memory (%s); '
                                'peaks will be NaN' % e)
            return
        with self._peaks_lock:
            if self._peak_private is None or private > self._peak_private:
                self._peak_private = private
            if self._peak_working_set is None or working_set > self._peak_working_set:
                self._peak_working_set = working_set

    def take_episode_peaks(self):
        """(peak private GB, peak working set GB) since the previous call, then reset.

        Samples once now too, so a window shorter than the interval still has a value.
        NaN when no sample could be read.
        """
        self._sample()
        with self._peaks_lock:
            private, working_set = self._peak_private, self._peak_working_set
            self._peak_private = self._peak_working_set = None
        return (math.nan if private is None else private / GB,
                math.nan if working_set is None else working_set / GB)
