import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = defaultdict(float)
        self.duration_sum = defaultdict(float)
        self.duration_count = defaultdict(int)
        self.gauges = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def gauge(self, name: str, value: float, **labels) -> None:
        key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
        with self._lock:
            self.gauges[key] = float(value)

    @contextmanager
    def timer(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self.duration_sum[name] += elapsed
                self.duration_count[name] += 1

    def prometheus(self) -> str:
        with self._lock:
            lines = []
            for name, value in sorted(self.counters.items()):
                lines.extend(["# TYPE diffjury_%s counter" % name, "diffjury_%s %s" % (name, value)])
            for name, value in sorted(self.duration_sum.items()):
                lines.extend([
                    "# TYPE diffjury_%s_seconds summary" % name,
                    "diffjury_%s_seconds_sum %s" % (name, value),
                    "diffjury_%s_seconds_count %s" % (name, self.duration_count[name]),
                ])
            typed = set()
            for (name, labels), value in sorted(self.gauges.items()):
                if name not in typed:
                    lines.append("# TYPE diffjury_%s gauge" % name)
                    typed.add(name)
                rendered = ",".join('%s="%s"' % (key, item.replace('"', "'")) for key, item in labels)
                lines.append("diffjury_%s%s %s" % (name, "{%s}" % rendered if rendered else "", value))
        return "\n".join(lines) + "\n"


metrics = Metrics()

