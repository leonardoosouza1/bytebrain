"""worker generico: puxa prosa de um MediaWiki (Wikipedia/Wikisource PT) com educacao + backoff."""
import sys, urllib.request, urllib.parse, json, time, re, os
HOST, OUT, MIN = sys.argv[1], sys.argv[2], float(sys.argv[3])
UA = "ByteBrain-research/1.0 (leonardo educational LM)"


def batch(n=8):
    url = f"https://{HOST}/w/api.php?action=query&format=json&generator=random&grnnamespace=0&grnlimit={n}&prop=extracts&explaintext=1&exlimit={n}&maxlag=5"
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=25)
    d = json.load(r); return [p.get("extract", "") for p in d.get("query", {}).get("pages", {}).values()]


def clean(t):
    t = re.sub(r"\n=+[^\n=]+=+\n", "\n", t)      # cabecalhos de secao == X ==
    t = re.sub(r"\n{3,}", "\n\n", t); return t.strip()


def main():
    t0 = time.time(); total = 0; delay = 6.0
    with open(OUT, "w", encoding="utf-8") as f:
        while time.time()-t0 < MIN*60:
            try:
                for ext in batch(8):
                    if len(ext) > 500:
                        c = clean(ext); f.write(c+"\n\n"); f.flush(); total += len(c.encode())
                delay = max(5.0, delay*0.9)
            except Exception as e:
                code = getattr(e, "code", 0)
                if code in (429, 503): delay = min(90.0, delay*2)
                print(f"[{HOST}] {e} -> delay {delay:.0f}s", flush=True)
            if int(time.time()-t0) % 30 < delay: print(f"[{HOST}] {total/1e6:.1f}MB | {time.time()-t0:.0f}s", flush=True)
            time.sleep(delay)
    print(f"[{HOST}] FIM {total/1e6:.1f}MB -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
