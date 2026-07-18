"""Tools the assistant can call during a response.

Each tool returns a plain string that goes back to the model as the tool
result. Keep results compact — they are LLM context, not user output.
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("tools")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Returns "
                           "top results with title, URL, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page and return its readable text "
                           "(truncated). Use after web_search for details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time (UTC).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _web_search_sync(query: str) -> str:
    from ddgs import DDGS
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}\n  {r.get('href', '')}\n"
                     f"  {r.get('body', '')[:300]}")
    return "\n".join(lines)


async def _fetch_url(url: str) -> str:
    import httpx
    from bs4 import BeautifulSoup
    async with httpx.AsyncClient(follow_redirects=True, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0"}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    return text[:4000] or "(page has no readable text)"


async def run_tool(name: str, arguments: str) -> str:
    """Execute a tool call; always returns a string (errors included)."""
    try:
        args = json.loads(arguments) if arguments and arguments.strip() else {}
    except json.JSONDecodeError as e:
        return f"error: could not parse tool arguments: {e}"
    try:
        if name == "web_search":
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _web_search_sync, str(args["query"])),
                timeout=20)
        if name == "fetch_url":
            return await _fetch_url(str(args["url"]))
        if name == "get_current_time":
            now = datetime.now(timezone.utc)
            return now.strftime("%A, %B %d, %Y, %H:%M UTC")
        return f"error: unknown tool {name!r}"
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"tool {name} failed: {e}")
        return f"error: {e.__class__.__name__}: {e}"
