"""Headline semantics: which sectors, which risk themes, how much shock language.

Deliberately rule-based and fully inspectable. Every tag on the dashboard can be traced to
a word list in this file, which matters more here than squeezing out recall: the layer is
unvalidated, so the least it owes the reader is transparency about what it is counting.

Three readings per headline:

* **sectors** — from an explicit ETF feed, a constituent company name/ticker, or sector
  vocabulary ("chipmaker" → Technology, "OPEC" → Energy).
* **themes**  — the risk channels volatility research keeps finding: policy, inflation,
  labour, earnings, geopolitics, trade, credit, politics, regulation, AI, energy.
* **shock**   — the volume of the language. Volatility is about magnitude, not direction:
  "stocks soar" and "stocks plunge" are both louder than "stocks edge higher".
"""

from __future__ import annotations

import re

# Constituents of the breadth universe (config.BREADTH_UNIVERSE) plus a few household
# names, mapped to their SPDR sector ETF. Aliases are matched as whole words.
COMPANIES: dict[str, list[str]] = {
    "XLK": ["apple", "microsoft", "nvidia", "intel", "cisco", "oracle", "ibm", "adobe",
            "salesforce", "qualcomm", "texas instruments", "amd", "broadcom", "micron",
            "palantir", "AAPL", "MSFT", "NVDA", "INTC", "CSCO", "ORCL", "ADBE", "CRM",
            "QCOM", "TXN", "AVGO", "MU"],
    "XLC": ["alphabet", "google", "meta", "facebook", "disney", "comcast", "verizon",
            "at&t", "netflix", "warner bros", "GOOGL", "GOOG", "META", "DIS", "CMCSA",
            "VZ", "NFLX"],
    "XLY": ["amazon", "tesla", "mcdonald's", "nike", "starbucks", "home depot", "lowe's",
            "ford", "general motors", "booking", "chipotle", "AMZN", "TSLA", "MCD", "NKE",
            "SBUX", "HD"],
    "XLF": ["jpmorgan", "jp morgan", "bank of america", "wells fargo", "citigroup", "citi",
            "goldman sachs", "goldman", "morgan stanley", "american express", "blackrock",
            "schwab", "berkshire", "visa", "mastercard", "JPM", "BAC", "WFC", "GS", "AXP",
            "BLK", "SCHW"],
    "XLV": ["johnson & johnson", "pfizer", "merck", "abbvie", "unitedhealth", "eli lilly",
            "lilly", "bristol myers", "amgen", "gilead", "cvs", "moderna", "novo nordisk",
            "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "BMY", "AMGN", "GILD"],
    "XLE": ["exxon", "exxonmobil", "chevron", "conocophillips", "schlumberger", "slb",
            "eog", "phillips 66", "occidental", "XOM", "CVX", "COP"],
    "XLP": ["procter & gamble", "coca-cola", "coke", "pepsico", "pepsi", "walmart",
            "costco", "target", "kraft heinz", "colgate", "philip morris", "PG", "KO",
            "PEP", "WMT", "COST", "TGT"],
    "XLI": ["boeing", "caterpillar", "general electric", "ge aerospace", "honeywell", "ups",
            "fedex", "union pacific", "3m", "lockheed", "lockheed martin", "deere",
            "raytheon", "rtx", "BA", "CAT", "HON", "UNP", "MMM", "LMT", "DE"],
    "XLU": ["nextera", "duke energy", "southern company", "dominion", "constellation energy",
            "NEE", "DUK", "AEP"],
    "XLB": ["linde", "air products", "sherwin-williams", "freeport", "freeport-mcmoran",
            "newmont", "dow inc", "dupont", "nucor", "LIN", "APD", "SHW", "FCX", "NEM"],
    "XLRE": ["prologis", "american tower", "equinix", "simon property", "PLD", "AMT", "EQIX"],
}

SECTOR_WORDS: dict[str, list[str]] = {
    "XLK": ["tech", "technology", "semiconductor", "semiconductors", "chip", "chips",
            "chipmaker", "chipmakers", "software", "cloud", "data center", "data centers",
            "cybersecurity", "nasdaq"],
    "XLC": ["telecom", "streaming", "social media", "advertising", "ad market", "media",
            "broadband", "wireless"],
    "XLY": ["retail", "retailer", "retailers", "consumer spending", "automaker",
            "automakers", "carmaker", "carmakers", "auto sector", "autos", "electric vehicle", "evs", "e-commerce", "holiday sales",
            "restaurant", "restaurants", "cruise", "hotel", "travel"],
    "XLF": ["bank", "banks", "banking", "lender", "lenders", "credit card", "insurer",
            "insurers", "fintech", "asset manager", "private credit", "brokerage"],
    "XLV": ["pharma", "pharmaceutical", "drugmaker", "drugmakers", "drug", "drugs",
            "biotech", "fda", "healthcare", "health care", "vaccine", "medicare",
            "medicaid", "hospital", "obesity drug", "glp-1"],
    "XLE": ["oil", "crude", "opec", "opec+", "natural gas", "brent", "wti", "refiner",
            "drilling", "shale", "gasoline", "lng"],
    "XLP": ["grocery", "groceries", "staples", "food prices", "beverage", "beverages",
            "household goods", "tobacco", "packaged food"],
    "XLI": ["airline", "airlines", "aerospace", "defense", "manufacturing", "freight",
            "railroad", "industrial", "industrials", "shipping", "factory", "factories",
            "trucking", "ism"],
    "XLU": ["utility", "utilities", "power grid", "electricity", "power demand",
            "nuclear power"],
    "XLB": ["mining", "miner", "miners", "gold", "copper", "steel", "chemicals",
            "aluminum", "lithium", "metals"],
    "XLRE": ["real estate", "reit", "reits", "housing", "home sales", "mortgage",
             "mortgage rates", "commercial property", "office space", "homebuilder",
             "homebuilders", "rent", "rents"],
}

THEMES: dict[str, list[str]] = {
    "monetary_policy": ["fed", "federal reserve", "fomc", "rate cut", "rate cuts",
                        "rate hike", "rate hikes", "interest rate", "interest rates",
                        "central bank", "ecb", "boj", "bank of japan", "bank of england",
                        "fed chair", "basis points", "dovish", "hawkish", "balance sheet"],
    "inflation": ["inflation", "cpi", "pce", "consumer prices", "price index",
                  "producer prices", "ppi", "deflation", "stagflation"],
    "labor": ["jobs report", "payrolls", "nonfarm", "unemployment", "jobless",
              "job openings", "layoffs", "labor market", "hiring", "job cuts", "wages"],
    "earnings": ["earnings", "quarterly results", "revenue", "guidance", "profit",
                 "profits", "eps", "outlook", "forecast", "beats", "misses", "q1", "q2",
                 "q3", "q4", "results"],
    "geopolitics": ["war", "missile", "missiles", "sanctions", "conflict", "invasion",
                    "military", "airstrike", "ukraine", "russia", "israel", "iran", "gaza",
                    "taiwan", "north korea", "ceasefire", "geopolitical"],
    "trade": ["tariff", "tariffs", "trade war", "trade deal", "export controls", "imports",
              "exports", "customs", "protectionism"],
    "credit": ["default", "defaults", "bankruptcy", "downgrade", "credit spread",
               "credit spreads", "debt ceiling", "bond yields", "treasury yields",
               "yields", "liquidity", "junk bonds", "private credit", "bond market"],
    "politics": ["election", "elections", "congress", "shutdown", "white house", "senate",
                 "house of representatives", "midterm", "midterms", "polls", "president",
                 "administration", "lawmakers"],
    "regulation": ["lawsuit", "antitrust", "sec", "probe", "investigation", "regulator",
                   "regulators", "fined", "settlement", "ban", "doj", "ftc", "court"],
    "ai": ["ai", "artificial intelligence", "chatbot", "openai", "anthropic", "llm",
           "generative ai", "gpu", "gpus"],
    "energy": ["oil", "opec", "crude", "gas prices", "energy prices", "brent"],
}

# Shock language. Weight 2 = unambiguous turbulence, weight 1 = elevated movement.
SHOCK_WORDS: dict[str, int] = {
    **{w: 2 for w in ["crash", "crashes", "plunge", "plunges", "plunged", "plummet",
                      "plummets", "rout", "collapse", "collapses", "meltdown", "crisis",
                      "panic", "bloodbath", "circuit breaker", "contagion", "freefall",
                      "capitulation", "black monday", "wipes out", "wiped out",
                      "worst day", "biggest drop", "biggest drop since"]},
    **{w: 1 for w in ["tumble", "tumbles", "sink", "sinks", "sank", "slump", "slumps",
                      "selloff", "sell-off", "sell off", "turmoil", "fear", "fears",
                      "volatility", "volatile", "swing", "swings", "spike", "spikes",
                      "soar", "soars", "surge", "surges", "jump", "jumps", "skyrocket",
                      "shock", "shocks", "whipsaw", "jitters", "brace", "warns", "warning",
                      "recession", "uncertainty", "slide", "slides", "tank", "tanks",
                      "record high", "record low", "turbulence", "vix", "best day",
                      "sharpest", "steepest"]},
}


def _compile(words: list[str], case_sensitive: bool = False) -> re.Pattern:
    alts = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(r"(?<![\w&])(" + "|".join(alts) + r")(?![\w&])", flags)


def _split(aliases: list[str]) -> tuple[list[str], list[str]]:
    """All-caps aliases are tickers: matched case-sensitively so 'T', 'DE', 'GE' or 'CAT'
    don't fire on ordinary words."""
    return [a for a in aliases if not a.isupper()], [a for a in aliases if a.isupper()]


_SECTOR_RX: dict[str, tuple[re.Pattern, re.Pattern | None]] = {}
for _etf in SECTOR_WORDS:
    _names, _tickers = _split(COMPANIES.get(_etf, []))
    _SECTOR_RX[_etf] = (_compile(_names + SECTOR_WORDS[_etf]),
                        _compile(_tickers, case_sensitive=True) if _tickers else None)
_THEME_RX = {k: _compile(v) for k, v in THEMES.items()}
_SHOCK_RX = _compile(list(SHOCK_WORDS))


def tag_sectors(text: str, hints: list[str] | None = None) -> list[str]:
    found = {h for h in (hints or []) if h in _SECTOR_RX}
    for etf, (words, tickers) in _SECTOR_RX.items():
        if words.search(text) or (tickers is not None and tickers.search(text)):
            found.add(etf)
    return sorted(found)


def tag_themes(text: str) -> list[str]:
    return sorted(k for k, rx in _THEME_RX.items() if rx.search(text))


def shock_score(text: str) -> int:
    return sum(SHOCK_WORDS[m.lower()] for m in _SHOCK_RX.findall(text))


def annotate(title: str, summary: str = "", hints: list[str] | None = None) -> dict:
    """Tag one headline. Title carries the weight; the summary only adds sector/theme
    context, never shock (summaries are wordier and would inflate it)."""
    text = f"{title}. {summary}" if summary else title
    return {"sectors": tag_sectors(text, hints), "themes": tag_themes(text),
            "shock": shock_score(title)}
