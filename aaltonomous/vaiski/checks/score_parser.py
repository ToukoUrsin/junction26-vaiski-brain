import json, sys
sys.path.insert(0, ".")
from aaltonomous.vaiski.language import parse
rows = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in rows if parse(r["text"]).mapping != r["mapping"]]
print(f"{len(rows)-len(bad)}/{len(rows)} correct")
for r in bad[:int(sys.argv[2]) if len(sys.argv)>2 else 40]:
    p = parse(r["text"])
    print("-", r["text"], "\n   want", r["mapping"], "\n   got ", p.mapping, p.pos, p.neg, p.notes)
