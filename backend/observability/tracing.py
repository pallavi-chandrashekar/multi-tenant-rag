"""OpenTelemetry tracing.

Provides distributed-tracing spans for the API request, retrieval, vector store,
and LLM stages plus token/latency attributes. Tracing is **off by default** and
gated on ``OTEL_ENABLED``; the ``opentelemetry-*`` packages are imported lazily
inside ``init_tracing`` so this module imports with only the standard library.

When tracing is disabled (or the packages are absent) ``span`` is a no-op context
manager with zero runtime overhead, so call sites stay unconditional:

    with span("rag.retrieval", tenant_id=tid) as s:
        ...
        set_attributes(s, retrieval_count=n, confidence=c)
"""

import contextlib

from backend.config import settings

_tracer = None
_enabled = False


def init_tracing(app=None) -> bool:
    """Initialise the tracer provider if ``OTEL_ENABLED`` and the SDK is present.

    Returns ``True`` when tracing was activated. Safe to call once at startup;
    failures degrade to disabled tracing rather than crashing the app.
    """
    global _tracer, _enabled
    if not settings.OTEL_ENABLED:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        print(
            "OTEL_ENABLED set but opentelemetry packages are not installed; "
            "tracing disabled. Install backend/requirements-otel.txt to enable."
        )
        return False

    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    exporter = None
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT
            )
        except ImportError:
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)
    _enabled = True

    # Optional FastAPI auto-instrumentation (per-request server spans).
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass

    print(f"OpenTelemetry tracing enabled (service={settings.OTEL_SERVICE_NAME}).")
    return True


@contextlib.contextmanager
def span(name: str, **attributes):
    """Context manager yielding a span (or ``None`` when tracing is disabled)."""
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        set_attributes(current, **attributes)
        yield current


def set_attributes(current, **attributes) -> None:
    """Set attributes on a span if tracing is active (no-op otherwise)."""
    if current is None:
        return
    for key, value in attributes.items():
        try:
            current.set_attribute(key, value)
        except Exception:  # noqa: BLE001 - observability must never break a request
            pass
