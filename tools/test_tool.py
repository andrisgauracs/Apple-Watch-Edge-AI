#!/usr/bin/env python3
"""Runs WatchLLM/Tools/Tools.json exactly as the app would, from your Mac.

Mirrors ToolBox in WatchLLM/Tools.swift: date-window substitution, optional
OAuth refresh, field extraction, ordered derived values, and the fact template.
Lets you check a tool config without rebuilding and reinstalling the app.
Prints results only — never credentials.
"""
import json, os, sys, urllib.parse, urllib.request, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG  = os.path.join(ROOT, "WatchLLM", "Tools", "Tools.json")

def post(url, form):
    body = urllib.parse.urlencode(form).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=15) as r:
        return json.load(r)

def get(url, headers):
    req = urllib.request.Request(url)
    for k, v in (headers or {}).items(): req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)

def resolve(node, path):
    for key in path:
        if key == "*":
            if isinstance(node, dict):  node = next(iter(node.values()), None)
            elif isinstance(node, list): node = node[0] if node else None
            else: return None
            if node is None: return None
            continue
        if isinstance(node, list):
            i = int(key)
            if i >= len(node): return None
            node = node[i]
        elif isinstance(node, dict):
            node = node.get(key)
        else:
            return None
        if node is None: return None
    return node

def evaluate(expr, nums):
    toks, cur = [], ""
    for ch in expr:
        if ch in "+-*/":
            toks.append(cur.strip()); cur = ""; toks.append(ch)
        else: cur += ch
    toks.append(cur.strip())
    toks = [t for t in toks if t]
    def val(t): return nums.get(t, None) if t not in nums else nums[t]
    acc = nums.get(toks[0], None)
    if acc is None:
        try: acc = float(toks[0])
        except ValueError: return None
    i = 1
    while i + 1 < len(toks):
        op, name = toks[i], toks[i+1]
        v = nums.get(name)
        if v is None:
            try: v = float(name)
            except ValueError: return None
        if op == "/":
            if v == 0: return None
            acc /= v
        elif op == "*": acc *= v
        elif op == "+": acc += v
        elif op == "-": acc -= v
        i += 2
    return acc

def fmt_raw(raw):
    """Mirror Swift's ToolBox.formatted(): group integers, leave anything else."""
    try:
        return f"{int(raw):,}"
    except (TypeError, ValueError):
        return str(raw)

def fmt_derived(v):
    """Mirror Swift: ratios read better with a decimal, counts grouped."""
    return f"{v:.1f}" if v < 100 else f"{int(round(v)):,}"

def main():
    user_query = sys.argv[1] if len(sys.argv) > 1 else ""
    specs = json.load(open(CFG))
    if not specs:
        print("Tools.json is empty ([]) — the app will run fully offline."); return 0
    matched = 0
    for spec in specs:
        if user_query and not any(t.lower() in user_query.lower() for t in spec["triggers"]):
            continue
        matched += 1
        # The app takes the FIRST matching spec in file order.
        tag = "  <- the app would use this" if matched == 1 and user_query else ""
        print(f"=== {spec['name']} ==={tag}")
        url = spec["url"]
        if "{query}" in url:
            topic = user_query.lower()
            for t in spec["triggers"]:
                if topic.startswith(t.lower()): topic = topic[len(t):]
            topic = topic.strip(" ?.!,")
            url = url.replace("{query}", urllib.parse.quote(topic, safe=""))
            print(f"  topic  : {topic!r}")
        if spec.get("windowDays"):
            end = datetime.datetime.now(datetime.timezone.utc).date()
            start = end - datetime.timedelta(days=spec["windowDays"])
            url = url.replace("{startDate}", start.isoformat()).replace("{endDate}", end.isoformat())
            print(f"  window: {start} .. {end}")
        headers = dict(spec.get("headers") or {})
        auth = spec.get("auth")
        if auth:
            form = {"client_id": auth["clientId"], "refresh_token": auth["refreshToken"],
                    "grant_type": "refresh_token"}
            if auth.get("clientSecret"): form["client_secret"] = auth["clientSecret"]
            try:
                tok = post(auth["tokenURL"], form)
            except urllib.error.HTTPError as e:
                print("  OAuth refresh FAILED:", e.code, e.read().decode()[:200]); return 1
            headers["Authorization"] = "Bearer " + tok["access_token"]
            print("  OAuth refresh: ok")
        try:
            data = get(url, headers)
        except urllib.error.HTTPError as e:
            print("  request FAILED:", e.code, e.read().decode()[:300]); return 1
        if "rows" in data:
            print("  raw rows:", data["rows"])
        text, nums = {}, {}
        for name, path in (spec.get("fields") or {}).items():
            node = resolve(data, path)
            if node is None:
                print(f"  MISSING field '{name}' at {path}"); return 1
            raw = str(node)
            lim = spec.get("maxChars")
            if lim and len(raw) > lim:
                cut = raw[:lim]
                raw = cut[:cut.rfind(".")+1] if "." in cut else cut
            text[name] = fmt_raw(raw)
            try: nums[name] = float(node)
            except (TypeError, ValueError): pass
        for d in (spec.get("derived") or []):
            v = evaluate(d["expr"], nums)
            if v is None: print(f"  derived '{d['name']}' could not be evaluated"); continue
            nums[d["name"]] = v; text[d["name"]] = fmt_derived(v)
        out = spec["factTemplate"]
        for k, v in text.items(): out = out.replace("{" + k + "}", v)
        print("  values :", {k: v for k, v in sorted(text.items())})
        print("  FACT   :", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
