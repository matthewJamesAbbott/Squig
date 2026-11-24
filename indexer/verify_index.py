#!/usr/bin/env python3
"""Spot-check the ddc-books Qdrant collection: counts, payload fields,
vector dims, and a sample semantic search. Usage:

    python verify_index.py                      # counts + dims + payload check
    python verify_index.py "how to ferment wine grapes"   # + sample search
"""
import json
import sys
import urllib.request

Q = "http://localhost:6333"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(Q + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())


def main():
    col = req("GET", "/collections/ddc-books")["result"]
    print(f"collection: ddc-books  vectors={col['status']}  "
          f"points={col['points_count']}  dims={col['config']['params']['vectors']['size']}  "
          f"distance={col['config']['params']['vectors']['distance']}")

    scroll = req("POST", "/collections/ddc-books/points/scroll",
                 {"limit": 3, "with_payload": True, "with_vectors": True})
    pts = scroll["result"]["points"]
    for p in pts:
        pl = p["payload"]
        print(f"\npoint id={p['id']}  vector_dim={len(p['vector'])}")
        print(f"  book_title={pl['book_title']!r}")
        print(f"  authors={pl['authors']!r}  publisher={pl['publisher']!r}  "
              f"pub_date={pl['pub_date']!r}  ddc={pl['ddc']!r}")
        print(f"  ddc_coarse={pl['ddc_coarse']!r}  ddc_tens={pl['ddc_tens']!r}  "
              f"page_num={pl['page_num']}  book_id={pl['book_id']}")
        print(f"  text={pl['text'][:110]!r}...")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"\n--- search: {q!r} ---")
        # Embed the query with the same model, then use the Qdrant >= 1.20
        # query API (the old /points/search endpoint was removed).
        ebody = json.dumps({"model": "bge-m3", "input": q}).encode()
        er = urllib.request.Request("http://localhost:11434/api/embed", data=ebody,
                                    headers={"Content-Type": "application/json"})
        qvec = json.loads(urllib.request.urlopen(er).read())["embeddings"][0]
        res = req("POST", "/collections/ddc-books/points/query",
                  {"query": qvec, "limit": 5, "with_payload": True})
        for hit in res["result"]["points"]:
            pl = hit["payload"]
            print(f"  {hit['score']:.4f}  {pl['book_title'][:60]!r} p.{pl['page_num']}  "
                  f"[{pl['ddc']}]  {pl['text'][:80]!r}")


if __name__ == "__main__":
    main()
