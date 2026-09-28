"""OpenTelemetry spans + Prometheus metrics around each pipeline stage's unit of work.

setup_telemetry() must be called once, early in any process that runs a stage, before
stage_span() is used. It starts a real prometheus_client HTTP server (scraped by a real
Prometheus server) and registers a console-exporting tracer, so spans are visible without
needing a separate trace backend.

Metric label cardinality matters: labels are "stage" (a handful of fixed values) and,
for failures, "error_type" (a bounded set of exception class names). input_hash is
deliberately a SPAN attribute only, never a metric label -- one label value per record
processed would be a cardinality explosion in Prometheus.

"Failure rate" isn't stored as its own metric -- it's the standard Prometheus pattern of
exposing the two counters (attempts, failures) and deriving the ratio in PromQL:
    sum(rate(pipeline_failures_total[5m])) by (stage)
  / sum(rate(pipeline_attempts_total[5m])) by (stage)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import start_http_server

SERVICE_NAME = "self-healing-pipeline"
DEFAULT_PROMETHEUS_PORT = 9464

_tracer = None
_rows_in = None
_rows_out = None
_attempts = None
_failures = None
_latency = None
_initialized = False


def setup_telemetry(prometheus_port: int = DEFAULT_PROMETHEUS_PORT) -> None:
    """Configure tracing + metrics and start the Prometheus HTTP server.

    Idempotent: only the first call in a process does anything, so it's safe to call
    from every entry point (a script, a test fixture, a long-running worker) without
    worrying about double-registering providers or double-binding the HTTP port.
    """
    global _tracer, _rows_in, _rows_out, _attempts, _failures, _latency, _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": SERVICE_NAME})

    trace.set_tracer_provider(TracerProvider(resource=resource))
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    _tracer = trace.get_tracer(SERVICE_NAME)

    # The SDK's default histogram buckets (0, 5, 10, 25 ... 10000) are calibrated
    # for a much coarser scale than these stages' actual latencies (milliseconds to
    # low single-digit seconds) -- nearly every sample would land in one bucket,
    # making quantiles meaningless interpolation. Use boundaries sized to what a
    # per-record clean, a batch embed, or a vectorstore load actually take.
    latency_view = View(
        instrument_name="pipeline_stage_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
        ),
    )

    reader = PrometheusMetricReader()
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[reader], views=[latency_view])
    )
    meter = metrics.get_meter(SERVICE_NAME)

    _rows_in = meter.create_counter(
        "pipeline_rows_in_total",
        unit="1",
        description="Rows/records/chunks presented to a stage",
    )
    _rows_out = meter.create_counter(
        "pipeline_rows_out_total",
        unit="1",
        description="Rows/records/chunks successfully produced by a stage",
    )
    _attempts = meter.create_counter(
        "pipeline_attempts_total",
        unit="1",
        description="Units of work attempted by a stage (denominator for failure rate)",
    )
    _failures = meter.create_counter(
        "pipeline_failures_total",
        unit="1",
        description="Units of work that failed in a stage",
    )
    _latency = meter.create_histogram(
        "pipeline_stage_duration_seconds",
        unit="s",
        description="Wall-clock time to process one unit of work in a stage",
    )

    start_http_server(prometheus_port)
    _initialized = True


class _SpanHandle:
    def __init__(self) -> None:
        self.rows_in = 1  # sensible default for a per-record stage; batches override it
        self.rows_out = 0
        self.failed_as: Optional[str] = None

    def set_rows_in(self, n: int) -> None:
        self.rows_in = n

    def set_rows_out(self, n: int) -> None:
        self.rows_out = n

    def mark_failed(self, error_type: str) -> None:
        """Record this unit of work as failed without raising an exception -- for
        stages that catch their own errors (to record them against the ledger) and
        continue, rather than letting the exception propagate out of the span.
        """
        self.failed_as = error_type


@contextmanager
def stage_span(stage: str, *, input_hash: Optional[str] = None):
    """Wrap one unit of work in a pipeline stage with a span + metrics.

    Usage:
        with stage_span("ingest", input_hash=h) as span:
            ... do the work ...
            span.set_rows_out(1)          # on success
            # or, on a caught (not re-raised) error:
            span.mark_failed(type(e).__name__)

    An exception that propagates out of the block is also treated as a failure (span
    gets the exception recorded, status ERROR, failure counter incremented with the
    exception's type name) and is then re-raised unchanged.
    """
    if not _initialized:
        raise RuntimeError("setup_telemetry() must be called before stage_span()")

    labels = {"stage": stage}
    _attempts.add(1, labels)
    handle = _SpanHandle()
    start = time.perf_counter()

    attrs = {"stage": stage}
    if input_hash is not None:
        attrs["input_hash"] = input_hash

    with _tracer.start_as_current_span(f"pipeline.{stage}", attributes=attrs) as span:
        try:
            yield handle
        except Exception as e:
            _rows_in.add(handle.rows_in, labels)
            _latency.record(time.perf_counter() - start, labels)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            _failures.add(1, {**labels, "error_type": type(e).__name__})
            raise
        else:
            _rows_in.add(handle.rows_in, labels)
            _latency.record(time.perf_counter() - start, labels)
            if handle.failed_as is not None:
                span.set_status(Status(StatusCode.ERROR, handle.failed_as))
                _failures.add(1, {**labels, "error_type": handle.failed_as})
            else:
                span.set_attribute("rows_out", handle.rows_out)
                _rows_out.add(handle.rows_out, labels)
