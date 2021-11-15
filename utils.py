import threading
import os


def round(a):
    return int(a) if a - int(a) < 0.5 else int(a) + 1


def run_task(items, target, args):
    """
    Splits items in several batches and runs target with args on each batch in parallel
    (on same number of threads as cpu cores)
    """
    chunk = len(items) // os.cpu_count()
    threads = []
    for startIndex in range(0, os.cpu_count()):
        if startIndex == os.cpu_count() - 1:
            filesToProcess = items[startIndex * chunk:]
        else:
            filesToProcess = items[startIndex * chunk: (startIndex + 1) * chunk]
        thread = threading.Thread(target=target, args=[filesToProcess] + args)
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()
