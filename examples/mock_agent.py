from nanoevals.types import Trace, ToolCall, UsageStats

RESPONSES = {
    "What is the weather in Tokyo?": Trace(
        input="What is the weather in Tokyo?",
        output="The weather in Tokyo is 22°C and sunny.",
        tool_calls=[ToolCall(name="get_weather", args={"city": "Tokyo"})],
        usage=UsageStats(latency_ms=120.0, input_tokens=15, output_tokens=25, cost=0.001),
    ),
    "Book a flight from NYC to Paris on June 15": Trace(
        input="Book a flight from NYC to Paris on June 15",
        output="Booked flight FL789 from NYC to Paris on June 15.",
        tool_calls=[
            ToolCall(name="search_flights", args={"origin": "NYC", "destination": "Paris", "date": "2026-06-15"}),
            ToolCall(name="book_flight", args={"flight_id": "FL789"}),
        ],
        usage=UsageStats(latency_ms=350.0, input_tokens=40, output_tokens=30, cost=0.003),
    ),
    "What restaurants are near the Eiffel Tower?": Trace(
        input="What restaurants are near the Eiffel Tower?",
        output="Here are restaurants near the Eiffel Tower: Le Jules Verne, Café de l'Homme.",
        tool_calls=[ToolCall(name="search_places", args={"query": "restaurants", "near": "Eiffel Tower"})],
        usage=UsageStats(latency_ms=200.0, input_tokens=20, output_tokens=35, cost=0.002),
    ),
    "Translate 'hello' to Japanese": Trace(
        input="Translate 'hello' to Japanese",
        output="こんにちは (Konnichiwa)",
        tool_calls=[ToolCall(name="translate", args={"text": "hello", "target_language": "Japanese"})],
        usage=UsageStats(latency_ms=90.0, input_tokens=12, output_tokens=10, cost=0.0005),
    ),
    "Cancel my hotel reservation in London": Trace(
        input="Cancel my hotel reservation in London",
        output="I tried to cancel but the system is unavailable right now.",
        tool_calls=[
            ToolCall(name="search_hotels", args={"city": "London"}),
            ToolCall(name="search_hotels", args={"city": "London", "retry": True}),
        ],
        usage=UsageStats(latency_ms=800.0, input_tokens=60, output_tokens=45, cost=0.005),
    ),
}

FALLBACK = Trace(
    input="",
    output="I don't understand that request.",
    tool_calls=[],
    usage=UsageStats(latency_ms=50.0, input_tokens=10, output_tokens=8, cost=0.0003),
)


def mock_agent(input_text: str) -> Trace:
    trace = RESPONSES.get(input_text, FALLBACK)
    if trace.input != input_text:
        trace = trace.model_copy(update={"input": input_text})
    return trace
