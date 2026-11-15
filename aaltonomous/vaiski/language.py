"""Instruction -> sorting rules, offline and deterministic (no network at the finals, no model, < 5 ms).

    parse("Sort the objects: red into right, blue into left, yellow into center.")
    -> Parse(mapping={'red': 'right', 'blue': 'left', 'yellow': 'center'}, confidence=0.97, ...)

    p.mapping            colour -> bin, or None = LEAVE ON THE COUNTER (told to, no rule, or too uncertain)
    p.target(col, shape) bin for one object; use this in the brain (handles shape rules, see below)
    p.rules              the organisers' scorer format (scene.resolve_bin): keys colour, shape, (colour, shape), "default"
    p.certainty          per key: (tag, estimated probability that the bin is right), tags below

Scoring is +2 right box, -1 wrong box, 0 on the counter, so a colour is sorted only when its box is more likely right
than not; anything else stays on the counter (see ABSTAIN_BELOW).

How it works
  1. Normalise: lower case, accents folded, multi-word phrases -> canonical tokens, typos repaired (edit distance 1
     on the six core words, missing spaces), ordinals / "between the other two" / "left of the middle one" resolved,
     words that only look like positions removed ("all right", "right now", "left on the counter", "left hand").
  2. Tokenise into COLOR / SHAPE / POS / NEG / ALL / EXCEPT / REST / LEAVE / FIX (self-correction) / separators.
  3. Clause pairing: split on strong (. ; ! ? newline) and weak (, : and while then) separators; inside a clause
     runs of colours and positions are paired (C P, P C, C C P, C C C P P P ...). Colours without a position carry
     into the next clause ("red and blue go left").
  4. Resolve per key: explicit evidence first (a later self-correction replaces what it corrects), then "the rest",
     "everything", negation ("neither left nor right"), relations ("left of the bin that gets blue"), the default
     rule when the instruction refers to it ("as usual but ..."), last elimination (two colours named, two different
     bins -> the third colour takes the third bin). A colour with no basis at all gets None.
  5. Shapes: "cubes left, cylinders middle, prisms right" gives rules by shape; "red cubes left, everything else
     right" gives (colour, shape) + default rules; shapes that are only asides ("the red ones, cubes or cylinders,
     go left") are ignored.
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass, field

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
FIXED = {"red": "left", "yellow": "center", "blue": "right"}

# Probability that the bin is right, per kind of evidence (measured on the stress set, see exp/language/REPORT.md).
CERTAINTY = {"explicit": 0.97, "rest": 0.95, "all": 0.93, "leave": 0.95, "usual": 0.9, "negelim": 0.9, "rel": 0.85,
             "rembin": 0.93, "inferred": 0.6, "unmentioned": 0.25, "conflict": 0.45, "guess": 0.3, "norule": 0.5, "unknown": 0.0}
# +2 / -1 scoring breaks even at 1/3; the margin covers miscalibration and the x1.5 weight of the unseen episode.
ABSTAIN_BELOW = 0.4
INFER_MISSING = True      # "red left, blue right" -> yellow center (the organisers' generator is always a bijection)

_COLOR_SYN = {
    "red": ["red", "reds", "reddish", "crimson", "scarlet", "ruby", "cherry", "maroon", "vermilion", "vermillion",
            "rose", "tomato", "brick", "carmine", "burgundy", "blood", "strawberry", "raspberry", "wine", "garnet",
            "punainen", "punaiset", "punaista", "rot", "rote", "roten", "roter", "rotes", "rotem", "rouge", "rouges",
            "rojo", "rojos", "roja", "rojas", "rosso", "rossi", "rossa", "rosse", "rod", "roda", "rott", "rode",
            "rood", "vermelho", "vermelhos", "vermelha", "vermelhas", "punane", "punased", "punast"],
    "yellow": ["yellow", "yellows", "yellowish", "gold", "golden", "lemon", "amber", "mustard", "canary", "banana",
               "sunflower", "honey", "butter", "daffodil", "sunny", "saffron", "blonde",
               "keltainen", "keltaiset", "keltaista", "gelb", "gelbe", "gelben", "gelber", "gelbes", "gelbem",
               "jaune", "jaunes", "amarillo", "amarillos", "amarilla", "amarillas", "giallo", "gialli", "gialla",
               "gialle", "gul", "gula", "gult", "gule", "geel", "gele", "amarelo", "amarelos", "amarela", "amarelas",
               "kollane", "kollased", "kollast"],
    "blue": ["blue", "blues", "bluish", "navy", "azure", "cobalt", "sapphire", "cyan", "indigo", "teal", "sky",
             "ultramarine", "turquoise", "violet", "purple", "lilac", "lavender", "cerulean", "denim", "ocean",
             "aqua", "marine", "periwinkle", "midnight",
             "sininen", "siniset", "sinista", "blau", "blaue", "blauen", "blauer", "blaues", "blauem", "bleu",
             "bleus", "bleue", "bleues", "azul", "azules", "azuis", "blu", "azzurro", "azzurri", "azzurra", "azzurre",
             "bla", "blaa", "blatt", "blauw", "blauwe", "sinine", "sinised", "sinist"],
}
_SHAPE_SYN = {
    "cube": ["cube", "cubes", "cubic", "cuboid", "cuboids", "box-shaped", "cube-shaped", "block-shaped", "kuutio", "kuutiot", "wurfel", "cubo", "cubos", "cubi", "kub", "kuber",
             "kuben", "kuberna", "kubus", "kubussen", "kuup", "kuubid"],
    "cylinder": ["cylinder", "cylinders", "cylindrical", "can-shaped", "tube-shaped", "tubes", "tube", "cans", "disc", "discs",
                 "disk", "disks", "puck", "pucks", "circular", "rollers", "sylinteri", "sylinterit", "lierio", "lieriot",
                 "zylinder", "cylindre", "cylindres", "cilindro", "cilindros", "cilindri", "cylindrar", "cylindern",
                 "cylindrarna", "cilinder", "cilinders", "silinder", "silindrid"],
    "triangle": ["triangle", "triangles", "triangular", "prism", "prisms", "wedge", "wedges", "three-sided",
                 "pyramid", "pyramids", "triangulars", "wedge-shaped", "kolmio", "kolmiot", "prisma", "prismat", "dreieck",
                 "dreiecke", "dreiecken", "prismen", "prisme", "prismes", "triangulo", "triangulos", "prismas",
                 "triangolo", "triangoli", "prismi", "triangel", "trianglar", "trianglarna", "prismor", "driehoek",
                 "driehoeken", "kolmnurk", "kolmnurgad"],
}
_POS_SYN = {
    "left": ["left", "leftmost", "left-most", "left-hand", "lefthand", "left-side", "leftward", "leftwards", "lefty",
             "port", "vasen", "links", "linke", "linken", "linker", "linkes", "linkse", "gauche", "izquierda",
             "izquierdo", "sinistra", "sinistro", "vanster", "vanstra", "esquerda", "esquerdo", "venstre",
             "westernmost", "west", "vasak", "vasakule", "vasakusse", "vasakpoolsesse"],
    "center": ["center", "centre", "central", "centered", "centred", "middle", "mid", "midmost", "in-between",
               "inbetween", "centermost", "centremost", "inner", "centers", "centres", "centerbin",
               "keski", "mitte", "mittlere", "mittleren", "mittlerer", "mittleres", "mittig", "milieu", "centro",
               "centrale", "medio", "mezzo", "mitten", "mellersta", "mittersta", "midden", "middelste", "meio",
               "midten", "keskmine", "keskmisse", "keskele"],
    "right": ["right", "rightmost", "right-most", "right-hand", "righthand", "right-side", "rightward", "rightwards",
              "starboard", "oikea", "rechts", "rechte", "rechten", "rechter", "rechtes", "rechtse", "droite",
              "derecha", "derecho", "destra", "destro", "hoger", "hogra", "direita", "direito", "hoyre",
              "easternmost", "east", "parem", "paremale", "paremasse", "parempoolsesse"],
}
# inflected languages: word prefix -> token (checked after the exact words above)
_STEMS = [("linker", ("P", "left")), ("rechter", ("P", "right")), ("middelst", ("P", "center")),
          ("punai", ("C", "red")), ("punas", ("C", "red")), ("keltai", ("C", "yellow")), ("kollas", ("C", "yellow")),
          ("sinise", ("C", "blue")), ("sinisi", ("C", "blue")), ("sinine", ("C", "blue")), ("sinis", ("C", "blue")),
          ("vasem", ("P", "left")), ("vasen", ("P", "left")), ("vasak", ("P", "left")),
          ("keskim", ("P", "center")), ("keskel", ("P", "center")), ("keskil", ("P", "center")),
          ("keskie", ("P", "center")), ("keskm", ("P", "center")),
          ("oikea", ("P", "right")), ("oikeal", ("P", "right")), ("parem", ("P", "right")),
          ("kuutio", ("H", "cube")), ("sylinter", ("H", "cylinder")), ("lierio", ("H", "cylinder")),
          ("kolmio", ("H", "triangle")), ("prisma", ("H", "triangle")), ("wurfel", ("H", "cube")),
          ("zylinder", ("H", "cylinder")), ("dreieck", ("H", "triangle")), ("triang", ("H", "triangle")),
          ("cylind", ("H", "cylinder")), ("cilind", ("H", "cylinder"))]
_NEG = {"not", "never", "no", "none", "nothing", "avoid", "without", "neither", "nor", "dont", "doesnt", "isnt",
        "arent", "shouldnt", "mustnt", "cannot", "cant", "wont", "ei", "ala", "eika", "eivat", "alkaa", "nicht",
        "kein", "keine", "keinen", "nie", "niemals", "pas", "jamais", "nunca", "ni", "non", "mai", "inte", "ej",
        "aldrig", "niet", "geen", "nooit", "avoids", "avoiding", "avoided", "barred", "forbidden", "forbid", "forbids",
        "prohibited", "prohibit", "prohibits", "banned", "ban", "bans", "reject", "rejects", "rejected", "refuse", "off-limits", "xnot", "disallowed", "nowhere",
        "kielletty", "verboten", "interdit", "prohibido", "vietato", "forbjudet"}
_EXCEPT = {"except", "excluding", "besides", "apart", "save", "paitsi", "ausser", "ausgenommen", "sauf", "excepte",
           "hormis", "excepto", "salvo", "tranne", "eccetto", "utom", "forutom", "behalve", "exception", "bar", "unless",
           "barring"}
_ALL = {"all", "every", "everything", "each", "any", "anything", "objects", "items", "things", "pieces", "kaikki",
        "jokainen", "alles", "alle", "tout", "tous", "toutes", "todo", "todos", "todas", "tutto", "tutti", "tutte",
        "alla", "allt", "whatever", "stuff"}
_SWAP = {"swap", "swapped", "swaps", "swapping", "switch", "switched", "switches", "exchange", "exchanged",
         "interchange", "interchanged", "interchanging", "trade", "trades", "traded", "trading", "exchanging",
         "flip", "flipped", "flips", "flipping", "switching", "transpose", "transposed", "vaihda", "vaihdetaan",
         "tausche", "tauschen", "getauscht", "byt", "byter"}
_REVERSE = {"reverse", "reversed", "reversal", "reversing", "opposite", "mirrored", "mirror", "mirroring",
            "mirror-image", "backwards", "inverted", "invert", "inverse", "kaanteinen", "umgekehrt", "spiegelverkehrt",
            "otherwayround", "viceversa"}
_USUAL = {"usual", "default", "normal", "standard", "regular", "original", "customary", "normally", "usually",
          "typical", "habitual", "tavallinen", "tavalliseen", "ublich", "ubliche", "vanlig", "vanliga"}
_REST = {"rest", "remaining", "remainder", "others", "else", "leftover", "leftovers", "other", "xrest", "loput",
         "muut", "ubrigen", "ubrige", "restlichen", "anderen", "resten", "ovriga", "reste", "autres", "restants",
         "restantes", "resto", "demas", "otros", "altri", "rimanenti", "overige", "andere"}
_WEAK = {"and", "while", "whereas", "then", "plus", "ja", "seka", "und", "sowie", "et", "y", "e", "ed", "och",
         "samt", "en", "ning", "whilst", "meanwhile", "so"}
_LEAVE = {"ignore", "ignored", "ignores", "ignoring", "skip", "skipped", "skips", "skipping", "exclude", "excluded",
          "omit", "omitted", "xleave", "ohita", "ignoriere", "ignorieren", "ignorera", "ignora", "ignorar", "ignorez"}
# weaker: "leave blue" = on the counter, but "leave blue as usual" / (in a default-relative instruction) = keep its bin
_KEEP = {"leave", "leaves", "leaving", "stay", "stays", "staying", "keep", "keeps", "keeping", "unchanged",
         "unaffected", "unaltered", "preserve", "preserved", "maintain", "retain", "retains",
         "jata", "jatetaan", "jaavat", "jaa", "lamna", "lamnas", "lass", "lasse", "lassen", "bleiben", "bleibt",
         "laisse", "laissez", "laisser", "restent", "deja", "dejar", "dejen", "quedan", "lascia", "lasciare",
         "restano", "laat", "blijven", "stanna", "stannar", "ligga", "liegen", "untouched", "alone", "xkeep"}
_KEEP_STRONG = {"unchanged", "unaffected", "unaltered", "preserve", "preserved", "maintain", "retain", "retains"}
_SOFT_ACT = {"touch", "touched", "change", "changed", "alter", "altered", "modify", "affect", "affected", "move", "moved"}
_CONTRAST = {"while", "whereas", "whilst", "meanwhile", "but", "mentre", "mientras", "tandis", "wahrend", "terwijl", "medan",
             "taas", "aber", "mais", "pero", "ma", "maar", "mutta", "men", "kuid"}
_ALL_NOUN = {"objects", "items", "things", "pieces", "stuff"}
_ACT = {"touch", "touched", "move", "moved", "sort", "sorted", "pick", "picked", "grab", "grabbed", "handle",
        "handled", "bother", "relocate", "relocated", "transfer", "transferred", "binned", "put", "place", "placed",
        "koske", "siirra", "rora", "anfassen", "bewegen", "toucher", "tocar", "toccare"}
_ONLY = {"only", "solely", "exclusively", "vain", "ainoastaan", "nur", "seulement", "solo", "endast", "bara"}
_TOO = {"too", "also", "likewise", "aswell", "myos", "auch", "aussi", "tambien", "anche", "ocksa", "ook"}
_BLOCK = {"block", "blocks", "square", "squares", "brick", "bricks", "boxes"}     # cube only next to other shapes
_ASIDE = {"whether", "regardless", "irrespective", "either"}
_BINW = (r"(?:bins?|box|boxes|containers?|crates?|baskets?|trays?|buckets?|tubs?|slots?|ones?|receptacles?|"
         r"compartments?|spots?|positions?|places?|piles?|bowls?)")
_COUNTER = r"(?:counter|table|mat|desk|surface|workspace|bench|tabletop|worktop)"

_LOOKUP = {}
for _c, _ws in _COLOR_SYN.items():
    for _w in _ws:
        _LOOKUP[_w] = ("C", _c)
for _s, _ws in _SHAPE_SYN.items():
    for _w in _ws:
        _LOOKUP.setdefault(_w, ("H", _s))
for _p, _ws in _POS_SYN.items():
    for _w in _ws:
        _LOOKUP[_w] = ("P", _p)
_CRE = "(?:" + "|".join(sorted((re.escape(w) for ws in _COLOR_SYN.values() for w in ws), key=len, reverse=True)) + ")"
_PRE = r"(?:left|right|center|centre|middle|leftmost|rightmost|central)"
_PCANON = {"left": "left", "leftmost": "left", "right": "right", "rightmost": "right", "center": "center",
           "centre": "center", "middle": "center", "central": "center"}

# ---- typo repair: only the six core words; real words one edit away are never touched
_FUZZY = {"yellow": ("C", "yellow"), "blue": ("C", "blue"), "left": ("P", "left"), "right": ("P", "right"),
          "center": ("P", "center"), "centre": ("P", "center"), "middle": ("P", "center")}
_FUZZY_STOP = set("""mentre entre ventre centra blume blut bluse luft fellow bellow mellow yell yelled yellowy below hello follow hollow pillow willow glue clue blur
    flue slue blew bile bale lube bluer blurb blunt bleed bled lift loft heft deft weft lest let lets eft cleft lent
    leaf felt self elf leg legs lefts night light might sight tight fight eight bight wight righto bright fright wright
    aright righe riga rift riot rigid ridge bring girth ought eighth weight height enter renter canter cent cents counter
    counters winter cinder caster carter cater carpenter central entered meddle muddle fiddle riddle diddle piddle
    middles mile mild module mingle mobile idle midi rightly rights write wrong ride side sides wide blues clue true
    due blade able table belt best rest test next text lets later letter better after""".split())
_SEG_WORDS = set("the to in into on go goes and is for of at side put all ones one objects stuff things items "
                 "bin box bins boxes container crate tray basket object thing item block blocks cube cubes".split())


def _edit1(a, b):
    """True if b is one insertion / deletion / substitution / adjacent transposition away from a."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b:
        return False
    if la == lb:
        d = [i for i in range(la) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _edit2_same_ends(a, b):
    """Two single-letter edits, same first and last letter (6-letter targets only): 'yeloow', 'cneter'."""
    if a[0] != b[0] or a[-1] != b[-1] or abs(len(a) - len(b)) > 1:
        return False
    for i in range(len(a) + 1):
        for cand in ({a[:i] + a[i + 1:]} if i < len(a) else set()) | {a[:i] + ch + a[i:] for ch in set(b)} | (
                {a[:i] + ch + a[i + 1:] for ch in set(b)} if i < len(a) else set()) | (
                {a[:i] + a[i + 1] + a[i] + a[i + 2:]} if i + 1 < len(a) else set()):
            if cand == b or _edit1(cand, b):
                return True
    return False


def _fuzzy(w):
    if w in ("rde", "erd", "redd"):
        return ("C", "red")
    if len(w) < 4 or w in _FUZZY_STOP or not w.isalpha():
        return None
    hits = {tok for t, tok in _FUZZY.items() if _edit1(w, t)}
    if not hits and len(w) >= 5:
        hits = {tok for t, tok in _FUZZY.items() if len(t) == 6 and _edit2_same_ends(w, t)}
    if not hits and len(w) >= 5 and w.endswith("s") and w[:-1] not in _FUZZY_STOP:      # "yelows", "bleus"
        hits = {tok for t, tok in _FUZZY.items() if tok[0] == "C" and (w[:-1] == t or _edit1(w[:-1], t))}
    return hits.pop() if len(hits) == 1 else None


def _side(w):
    if w in ("left", "right"):
        return w
    f = _fuzzy(w)
    return f[1] if f and f[0] == "P" else None


def _segment(w):
    """'redobjects' / 'leftbin' / 'bluebox' -> tokens, when every part is a known word and one is a colour/position."""
    if len(w) < 6 or not w.isalpha() or w in ("redone", "blueprint", "leftovers", "rightly"):
        return None
    for i in range(2, len(w) - 1):
        a, b = w[:i], w[i:]
        ta, tb = _LOOKUP.get(a), _LOOKUP.get(b)
        if (ta or a in _SEG_WORDS) and (tb or b in _SEG_WORDS) and (ta or tb):
            if (ta and ta[0] in "CP" and len(a) >= 3) or (tb and tb[0] in "CP" and len(b) >= 3):
                return [t for t in (ta, tb) if t]
    return None


@dataclass
class Parse:
    mapping: dict                                   # colour -> bin | None (None = leave on the counter)
    confidence: float
    pos: list = field(default_factory=list)         # (key, bin) positive evidence
    neg: list = field(default_factory=list)         # (key, bin) negative evidence
    notes: list = field(default_factory=list)
    rules: dict = field(default_factory=dict)       # organisers' format; keys colour / shape / (colour, shape) / "default"
    certainty: dict = field(default_factory=dict)   # key -> (tag, probability)
    by_shape: bool = False                          # True when the bin depends on the object's shape
    raw: dict = field(default_factory=dict)         # best guess per colour before abstaining

    def target(self, color, shape=None):
        """Bin for one object, or None = leave it on the counter. shape: cube | cylinder | triangle | lying | None
        (lying = a prism or cylinder on its side; unknown shapes are sorted only when the shape does not matter)."""
        if not self.by_shape:
            return self.rules.get(color)
        if shape in SHAPES:
            return _resolve_rule(self.rules, color, shape)
        cands = ("cylinder", "triangle") if shape == "lying" else SHAPES
        got = {_resolve_rule(self.rules, color, s) for s in cands}
        return got.pop() if len(got) == 1 else None

    def table(self):
        """{'red/cube': bin | None, ...} for all 9 object types."""
        return {f"{c}/{s}": self.target(c, s) for c in COLORS for s in SHAPES}

    def __repr__(self):
        m = ", ".join(f"{c}->{self.mapping.get(c)}" for c in COLORS)
        r = f"; rules={self.rules}" if self.by_shape else ""
        return f"Parse({m}; conf={self.confidence:.2f}{r}; pos={self.pos} neg={self.neg} {self.notes})"


def executed(p: "Parse", task=None) -> dict:
    """colour -> bin | None (= leave on the counter) the robot should act on; None also when the bin depends on the
    shape (then ask p.target(colour, shape) per object). On the fixed rung the rule is known: an instruction only
    replaces it when every colour is read with full certainty."""
    if task == "sort_fixed":
        sure = not p.by_shape and all(p.rules.get(c) and p.certainty.get(c, ("", 0.0))[1] >= 0.9 for c in COLORS)
        return {c: p.rules[c] for c in COLORS} if sure else dict(FIXED)
    if p.by_shape:
        return {c: (lambda t: t.pop() if len(t) == 1 else None)({p.target(c, s) for s in SHAPES}) for c in COLORS}
    return {c: p.rules.get(c) for c in COLORS}


def _resolve_rule(rules, color, shape):
    for k in ((color, shape), color, shape, "default"):
        if k in rules:
            return rules[k]
    return None


_ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": 2, "final": 2, "1": 0, "2": 1,
        "3": 2, "one": 0, "two": 1, "three": 2, "a": 0, "b": 1, "c": 2}


_CNT = (r"\b(?:count(?:ing|ed|s)?|number(?:ed|ing|s)?|start(?:ing|s)?|begin(?:ning|s)?|reading|ordered|order|indexed)"
        r"\s+(?:\w+\s+){0,3}?from\s+(?:the\s+|your\s+|my\s+)?(?:far\s+)?(left|right)(?:most)?(?:\s+(?:side|end|edge))?\b")


def _ordinals(t: str) -> str:
    """'second bin from the right' -> center; 'the first box' -> left; 'bin 3' -> right; 'counting from the left: blue
    first, yellow second' -> bins. Ordinals that do not name a bin ('first, put ...') are removed as sequence words."""
    m = re.search(_CNT, t)
    ctx = bool(m) or bool(re.search(r"\bleft\s+to\s+right\b|\bright\s+to\s+left\b", t))
    flip = bool(m and m.group(1) == "right") or bool(re.search(r"\bright\s+to\s+left\b", t))
    t = re.sub(_CNT, lambda m: " xorderlr " if m.group(1) == "left" else " xorderrl ", t)

    has_first = bool(re.search(r"\b(?:first|1st)\s+(?:orange\s+)?" + _BINW + r"\b", t))

    def rep(m):
        if m.group(1) in ("last", "final") and not m.group(3):
            return " xlastbin " if not (flip or ctx or has_first) else (" left " if flip else " right ")
        i = _ORD[m.group(1)]
        side = m.group(3)
        if side == "right" or (not side and flip):
            i = 2 - i
        return " " + BINS[i] + " " + (m.group(2) or "") + " "
    num = lambda m: " " + BINS[2 - _ORD[m.group(1)] if flip else _ORD[m.group(1)]] + " bin "    # noqa: E731
    counted = bool(re.search(r"\b(?:first|1st|second|2nd|third|3rd)\s+(?:orange\s+)?" + _BINW + r"\b", t))
    t = re.sub(r"\b(first|1st|second|2nd|third|3rd|last|final)\s+(?:orange\s+)?(" + _BINW + r")?\s*"
               r"(?:(?:counting|counted|seen|as\s+seen|when\s+counting)\s+)?from\s+(?:the\s+|your\s+)?(left|right)\b", rep, t)
    t = re.sub(r"\b(first|1st|second|2nd|third|3rd|last|final)\s+(?:orange\s+)?(" + _BINW + r")\b()", rep, t)
    t = re.sub(r"\b(?:bin|box|container|crate|basket|tray|bucket|tub|slot|position)\s*(?:number\s*|no\.?\s*|nr\.?\s*|#\s*)?"
               r"(1|2|3|one|two|three)\b(?!\s+(?:steps?|places?|positions?|spots?|slots?|over|further|farther|more|along|across)\b)", num, t)
    t = re.sub(r"\b(?:number|no\.|nr\.|#)\s*(1|2|3|one|two|three)\b", num, t)
    # bare ordinals next to a colour or after "the": bins when the text counts the bins, else sequence words
    pat = (r"(\bthe\s+|\b" + _CRE + r"(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?|stuff))?\s+(?:goes\s+|go\s+|is\s+|are\s+|"
           r"comes?\s+)?)(first|1st|second|2nd|third|3rd)\b(?!\s+(?:of\s+all|thing|step|time|rule|round|trial|test|task|priority|and\s+foremost))")
    if ctx or counted or len({_ORD[m.group(2)] for m in re.finditer(pat, t)}) >= 2:
        t = re.sub(pat, lambda m: m.group(1) + " " + BINS[2 - _ORD[m.group(2)] if flip else _ORD[m.group(2)]] + " bin ", t)
        t = re.sub(r"\b(?:in|into|to|in\s+to)\s+the\s+(?:very\s+)?(?:last|final)(?:\s+one)?\b(?!\s+(?:of\s+all|thing|step|time|rule|round|"
                   r"trial|test|task|place|moment|second|minute))", " left bin " if flip else " right bin ", t)
    if ctx or counted:
        t = re.sub(r"\b(?:in|into|to)\s+(?:number\s+)?(one|two|three)\b(?=\s*[.,;!?\n]|\s*$)", num, t)
    t = re.sub(r"\b(?:first|1st|second|2nd|third|3rd|last|firstly|secondly|thirdly|lastly|finally)\b", " ", t)
    return t


def _positions(t: str) -> str:
    """Descriptive positions -> left / center / right."""
    bw = r"(?:\s+" + _BINW + r")?"
    t = re.sub(r"\b(left|right|center|centre|middle)\s+(?:one\s+)?(?:of|among|out\s+of)\s+(?:those|these|the|all\s+the|all)\s+"
               r"(?:three\s+|3\s+|orange\s+)*(?:bins|boxes|containers|crates|baskets|trays|buckets|tubs|slots)\b", r" \1 ", t)
    t = re.sub(r"\bbetween\s+(?:the\s+)?(?:other\s+two|others|two\s+others|outer\s+\w+|two\s+outer\s+\w+|two\s+end\s+\w+|"
               r"ends|outside\s+\w+|two\s+side\s+\w+|side\s+\w+|left\s+and\s+(?:the\s+)?right|right\s+and\s+(?:the\s+)?left)"
               + bw, " center ", t)
    t = re.sub(r"\b(?:one|bin|box|container|crate|basket|tray)\s+(?:in\s+|in-)?between\b(?!\s+(?:the\s+)?" + _CRE + r")(?:\s+(?:them|those|these|"
               r"the\s+two|the\s+two\s+ends|both))?", " center ", t)
    t = re.sub(r"\b(?:in\s+|in-)?between\s+(?:them|those\s+two|those|these\s+two|these|the\s+two\s+ends|the\s+two|both)\b", " center ", t)
    t = re.sub(r"\b(?:left|right)\s+of\s+(?:the\s+)?(?:leftmost|rightmost|left|right)(?:-hand)?(?:\s+" + _BINW + r")?(?:\s+itself)?", " center ", t)
    t = re.sub(r"\b(?:center|centre|middle)\s+of\s+(?:the\s+)?(?:other\s+two|others|two\s+others|row|line|three|lineup|line-up|trio|"
               r"other\s+" + _BINW + r"|two\s+other\s+" + _BINW + r"|other\s+two\s+" + _BINW + r")", " center ", t)
    t = re.sub(r"\b(?:the\s+)?(?:opposite|other)\s+(?:end|side|edge|extreme)\b", " xopp ", t)
    t = re.sub(r"\bnot\s+(?:at|on)\s+(?:either|any|both|an|the)\s+(?:end|ends|side|sides|edge|edges)\b", " center ", t)
    t = re.sub(r"\b(?:neither|not an?)\s+(?:end|edge|outer)\b", " center ", t)

    def third(m):
        a, b = _PCANON[m.group(1)], _PCANON[m.group(2)]
        rest = [x for x in BINS if x not in (a, b)]
        return (" " + rest[0] + " ") if len(rest) == 1 else m.group(0)
    t = re.sub(r"\bneither\s+(?:in\s+|on\s+|to\s+)?(?:the\s+)?(" + _PRE + r")" + bw + r"\s+nor\s+(?:in\s+|on\s+|to\s+)?(?:the\s+)?("
               + _PRE + r")" + bw, third, t)
    # "immediately to the right of the middle one" -> right
    t = re.sub(r"\b(left|right)\s+of\s+(?:the\s+)?(?:middle|center|centre|central)" + bw, r" \1 ", t)
    t = re.sub(r"\b(?:next\s+to|beside|adjacent\s+to|neighbou?ring)\s+the\s+(?:left|right)(?:most|-hand)?\s+" + _BINW,
               " center ", t)
    return t


def _relations(t: str) -> str:
    """Bins named through another colour: 'left of the bin that gets blue', 'same bin as red', 'where blue normally goes'."""
    gap = r"(?:(?!" + _CRE + r"\b|" + _PRE + r"\b|it\b|them\b)[a-z']+\s+){0,6}?"
    noun = r"(?:\s+(?:objects?|ones?|items?|things?|pieces?|blocks?|stuff|colou?r))?"
    dest = r"(?:" + _BINW[3:-1] + r"|destinations?|locations?|targets?|homes?|assignments?)"
    t = re.sub(r"\b(" + _CRE + r")" + noun + r"(?:'s|s')\s+(?:(?!" + _CRE + r"\b)[a-z]+\s+){0,2}?" + dest + r"\b",
               lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b(" + _CRE + r")'s(?![a-z])", lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\bwhere\s+(?:the\s+)?(" + _CRE + r")(?:\s+\w+)?\s+(?:usually|normally|typically|ordinarily|generally|"
               r"would|used|otherwise|always)\b(?:\s+(?:go|goes|went|belong|belongs|sit|sits|live|lives|to\s+go))?",
               lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b" + _BINW + r"\s+(?:that\s+|which\s+)?(?:usually|normally|typically|ordinarily)\s+(?:\w+\s+){0,2}?"
               r"(?:the\s+)?(" + _CRE + r")\b(?:\s+(?:ones|objects|items|things|pieces|blocks))?",
               lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b" + _BINW + r"\s+(?:that\s+|which\s+|where\s+)?(?:the\s+)?(" + _CRE + r")(?:\s+(?:ones|objects|items|things|pieces|blocks))?"
               r"\s+(?:usually|normally|typically|ordinarily|always|would|used)\b(?:\s+(?:gets?|takes?|uses?|goes|go|lands?|has|have|"
               r"occupy|occupies|belongs?|sits?|lives?|to))*", lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b(?:usual|normal|default|regular|standard|original|customary)\s+" + dest + r"\s+(?:of|for|from|belonging\s+to)\s+"
               r"(?:the\s+)?(" + _CRE + r")\b(?:\s+(?:ones|objects|items|things|pieces|blocks))?",
               lambda m: f" xdef{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b(left|right)\s+of\s+" + gap + r"(" + _CRE + r")\b(?:\s+(?:ones?|objects?|items?|things?|pieces?|"
               r"blocks?|" + _BINW + r"))?(?:\s+(?:go|goes|is|are|sits?|lands?|ends?\s+up))?",
               lambda m: f" xrel{m.group(1)[0]}{_LOOKUP[m.group(2)][1]} ", t)
    t = re.sub(r"\b(?:next\s+to|beside|adjacent\s+to|neighbou?ring|alongside)\s+" + gap + r"(" + _CRE + r")\b"
               r"(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?|" + _BINW + r"))?",
               lambda m: f" xreln{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\bsame\s+" + _BINW + r"\s+(?:as|that|where|with)\s+" + gap + r"(" + _CRE + r")\b"
               r"(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?))?(?:\s+(?:go|goes|went|do|does|is|are))?",
               lambda m: f" xsame{_LOOKUP[m.group(1)][1]} ", t)
    def _tw(m):                                    # "put red together with yellow to the right": one list, one bin
        before = re.split(r"[.,;:!?\n]", t[:m.start()])[-1]
        return m.group(0) if re.search(r"\b" + _PRE + r"\b", before) else f" and {m.group(1)} "
    t = re.sub(r"\b(?:joins?|joining|(?:together|along)\s+with)\s+(?:the\s+|all\s+the\s+|all\s+)?(" + _CRE + r")\b(?=[^.,;:!?\n]*\b" + _PRE + r"\b)", _tw, t)
    t = re.sub(r"\b(?:join|joins|joining|accompany|accompanies)\s+" + gap + r"(" + _CRE + r")\b(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?))?",
               lambda m: f" xsame{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"\b(?:together|along)\s+with\s+" + gap + r"(" + _CRE + r")\b(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?))?",
               lambda m: f" xsame{_LOOKUP[m.group(1)][1]} ", t)
    t = re.sub(r"(?<!is )(?<!are )(?<!be )(?<!'s )(?<!thats )(?<!remember )(?<!know )(?<!see )(?<!guess )(?<!recall )(?<!forget )\bwhere(?:ver)?\s+(?:the\s+)?(" + _CRE + r")(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?))?\s+"
               r"(?:go|goes|went|is|are|do|does|did)\b(?!\s+(?:to\b|supposed|meant|placed|put|going|sorted|deposited|dropped|headed|"
               r"sent|destined|collected|kept|stored))", lambda m: f" xsame{_LOOKUP[m.group(1)][1]} ", t)
    # bins named relative to the previously named bin: "the bin just right of it", "the one beside it"
    it = r"(?:it|that|that\s+one|this\s+one|the\s+previous\s+one|the\s+last\s+one)\b"
    t = re.sub(r"\btwo\s+(?:" + _BINW + r"|steps|places|positions|spots)\s+(?:over\s+)?(?:to\s+the\s+)?(?:left|right)\s+of\s+" + it, " xopp ", t)
    t = re.sub(r"\b(left|right)\s+of\s+" + it, lambda m: f" xadj{m.group(1)[0]} ", t)
    t = re.sub(r"\b(?:one|1|a)\s+(?:" + _BINW + r"|step|place|position|spot)\s+(?:over|along|across|further|farther|on)\s+from\s+" + it +
               r"(?:\s*,?\s*(?:toward|towards|to|on)\s+(?:your|the|my)\s+(left|right))?", lambda m: f" xadj{(m.group(1) or 'n')[0]} ", t)
    t = re.sub(r"\bto\s+(?:its|their)\s+(?:immediate\s+)?(left|right)\b", lambda m: f" xadj{m.group(1)[0]} ", t)
    t = re.sub(r"\b(?:next\s+to|beside|adjacent\s+to|neighbou?ring|alongside|after)\s+" + it, " xadjn ", t)
    t = re.sub(r"\bbefore\s+" + it, " xadjl ", t)
    t = re.sub(r"\b(?:next|following|neighbou?ring|adjacent)\s+" + _BINW + r"\s+(?:over|along|across)\b", " xadjn ", t)
    t = re.sub(r"\b(?:the\s+)?(?:very\s+)?same\s+" + _BINW + r"\b", " xsamebin ", t)
    return t


_PHRASES = (("left hand", "left-hand"), ("right hand", "right-hand"), ("left most", "leftmost"),
            ("right most", "rightmost"), ("far left", "leftmost"), ("far right", "rightmost"),
            ("left side", "left-side"), ("right side", "right-side"), ("in between", "in-between"),
            ("van links naar rechts", "left to right"), ("von links nach rechts", "left to right"),
            ("de gauche a droite", "left to right"), ("de izquierda a derecha", "left to right"),
            ("da sinistra a destra", "left to right"), ("vasemmalta oikealle", "left to right"),
            ("fran vanster till hoger", "left to right"), ("vasakult paremale", "left to right"),
            ("other way round", "otherwayround"), ("other way around", "otherwayround"),
            ("the other way", "otherwayround"), ("vice versa", "viceversa"), ("in that order", "respectively"),
            ("in this order", "respectively"), ("in order", "respectively"),
            ("in that very same order", "respectively"), ("in that same order", "respectively"),
            ("in the same order", "respectively"), ("in the order listed", "respectively"),
            ("in the order given", "respectively"),
            ("same as before", "usual"), ("as always", "usual"), ("as usual", "usual"), ("like always", "usual"),
            ("what you always do", "usual"), ("as you always do", "usual"), ("business as usual", "usual"),
            ("where it always", "usual"), ("where they always", "usual"), ("as before", "usual"),
            ("trade places", "swap"), ("change places", "swap"), ("switch places", "swap"), ("swap places", "swap"),
            ("as is", "unchanged"), ("as it is", "unchanged"), ("as they are", "unchanged"),
            ("other than", "except"), ("apart from", "except"), ("aside from", "except"),
            ("with the exception of", "except"), ("bis auf", "except"), ("lukuun ottamatta", "except"),
            ("a l'exception de", "except"), ("a l'exception des", "except"), ("a excepcion de", "except"),
            ("as well", "aswell"),
            ("anywhere but", "except"), ("nowhere but", "except"), ("nothing but", "except"), ("anywhere else", "xelse"),
            ("rather than", "not"), ("instead of", "not"), ("as opposed to", "not"),
            ("sky blue", "blue"), ("light blue", "blue"), ("dark blue", "blue"), ("dark red", "red"),
            ("light red", "red"), ("bright red", "red"), ("bright yellow", "yellow"), ("navy blue", "blue"),
            ("royal blue", "blue"), ("fire engine", "red"), ("fire-engine", "red"), ("on the left", "left"),
            ("on the right", "right"),
            ("in the middle", "center"), ("to the left", "left"), ("to the right", "right"),
            ("left one", "left"), ("right one", "right"), ("middle one", "center"), ("center one", "center"),
            ("centre one", "center"), ("triangular prism", "prism"), ("triangular prisms", "prisms"),
            ("three sided", "three-sided"), ("box shaped", "box-shaped"), ("can shaped", "can-shaped"),
            ("do nothing with", "xleave"), ("hands off", "xleave"), ("left over", "xrest"), ("left-over", "xrest"),
            ("everything else", "xrest"), ("anything else", "xrest"), ("all else", "xrest"),
            ("all the others", "xrest"), ("all others", "xrest"), ("all other", "xrest"), ("any other", "xrest"),
            ("every other", "xrest"), ("the other ones", "xrest"), ("what remains", "xrest"),
            ("whatever remains", "xrest"), ("alla andra", "xrest"), ("kaikki muut", "xrest"),
            ("alle anderen", "xrest"), ("tous les autres", "xrest"), ("todos los demas", "xrest"),
            ("tutti gli altri", "xrest"), ("no matter", "regardless"), ("irrespective of", "regardless of"))

_FIXRE = (r"\b(?:no\s*,?\s*wait|wait\s*,?\s*no|no\s*,?\s*sorry|sorry|scratch\s+that|strike\s+that|i\s+mean|i\s+meant|"
          r"actually|correction|oops|whoops|my\s+mistake|my\s+bad|make\s+that|make\s+it|on\s+second\s+thought|"
          r"i\s+misspoke|let\s+me\s+correct|let\s+me\s+rephrase|change\s+of\s+plan|hold\s+on|hang\s+on|or\s+rather|rather|"
          r"er+m?\s*,?\s*no|u[hm]+\s*,?\s*no|no\s*,?\s*no|nope|i\s+take\s+that\s+back|forget\s+that|ignore\s+that|"
          r"wait|update|edit|eikun|ei\s+kun|ei\s*,?\s*vaan|siis|anteeksi|nein|falsch|quatsch|korrektur|entschuldigung|"
          r"pardon|non\s*,|nej\s*,|forlat|perdon|scusa|sorry\s*,)\b")


def _fold(text: str) -> str:
    t = text.lower().replace("ß", "ss").replace("ø", "o").replace("æ", "ae")
    for a in ("≠", "≢", "∉", "⊄"):
        t = t.replace(a, " not ")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    for a in "’‘`´":
        t = t.replace(a, "'")
    for a in "—–―":
        t = t.replace(a, " , ")
    for a in ("⇄", "↔", "⇆", "⇋", "<->", "<=>", "<-->"):
        t = t.replace(a, " swap ")
    for a in ("≠", "!=", "=/=", "<>", "✗", "✘", "❌", "🚫"):
        t = t.replace(a, " not ")
    t = re.sub(r"\b(" + _CRE + r")\s*(?:→|⇒|➜|➔|⟶|➡|->|=>)\s*(" + _CRE + r")\b", r"\1 \2", t)
    for a in ("→", "⇒", "➜", "➔", "⟶", "➡", "->", "=>", "←", "⇐", "<-", "<="):
        t = t.replace(a, " xarrow ")
    t = t.replace("…", " , ").replace("...", " , ").replace("•", "\n").replace("·", "\n").replace("|", "\n").replace("/", " / ")
    return t


def _normalise(text: str) -> str:
    t = " " + _fold(text) + " "
    t = re.sub(r"(?<=[a-z0-9])\(\s*(" + _CRE + r")\s*\)", r" \1 ", t)       # "destination(red) = left bin"
    # parentheticals without a colour or position are asides ("(whether a cube or prism)")
    t = re.sub(r"\(([^)]*)\)", lambda m: (" , " + m.group(1) + " , ") if re.search(r"\b" + _CRE + r"\b", m.group(1))
               and re.search(r"\b" + _PRE + r"\b", m.group(1)) else " ", t)
    t = re.sub(r"(\w)n't\b", r"\1 not", t)            # doesn't -> does not
    t = re.sub(r"\bin\s+(?:exactly\s+|precisely\s+|just\s+|strictly\s+)?(?:that|this|the\s+same|the\s+very\s+same|same)\s+"
               r"(?:exact\s+|very\s+|precise\s+|same\s+)*(?:order|sequence)\b", " respectively ", t)
    t = re.sub(r"(?<!in )\border\s*:", " in order : ", t)
    t = t.replace("can't", "cannot").replace("won't", "will not").replace("cannot", "can not")
    t = re.sub(r"\bfrom\s+(?:your|the|my|its)\s+(left|right)(?:\s+(?:side|end|edge))?\s+to\s+(?:your|the|my|its)\s+(left|right)\b"
               r"(?:\s+(?:side|end|edge)\b)?(?!\s*-?\s*(?:hand\s+)?" + _BINW + r"\b)",
               lambda m: m.group(0) if m.group(1) == m.group(2) else f" from {m.group(1)} to {m.group(2)} ", t)
    t = re.sub(r"\bxarrow\s+(?:none|nothing|nowhere|no\s+bin|no\s+box|skip|null|n\s*/\s*a)\b", " xleave ", t)
    t = re.sub(r"\b(?:do\s+not|dont|never|no\s+need\s+to|without)\s+(?:swap|switch|flip|mirror|revers|invert|exchang|interchang|"
               r"mix)\w*\b(?:\s+(?:the|them|it|up|around|any|anything|things|two|both|bins?|boxes|colou?rs?|positions?|"
               r"places?|sides?|order|objects?|ones?)\b)*(?:\s+(?:" + _PRE + "|" + _CRE + r")\s+(?:and|with|or)\s+(?:the\s+)?(?:"
               + _PRE + "|" + _CRE + r")\b)?", " ", t)
    t = re.sub(r"\b(?:that's|thats|that\s+is|that\s+was|this\s+is|it's|it\s+is)\s+(?:not\s+right|not\s+correct|wrong|incorrect|"
               r"a\s+mistake)\b(?:\s+at\s+all)?", " xfix ", t)
    t = re.sub(r"\b(?:(?:do|does|did|will|would|should|must|can|could|shall|wo|ca)\s+not|never|no\s+need\s+to|without|not)\s+(?:forget|"
               r"hesitate|hesitating|hesitation|fail|neglect|worry|mind|stress|panic|rush|hurry|delay|fret|be\s+afraid|be\s+shy|doubt)\b"
               r"(?!\s+(?:about\s+)?(?:the\s+|its\s+|their\s+)?(?:shapes?|colou?rs?)\b)(?:\s+(?:to|that|about|it))?", " ", t)
    t = re.sub(r"\b(?:in\s+other\s+words|on\s+the\s+(?:other|one)\s+hand|each\s+other|one\s+another|put\s+another\s+way|"
               r"the\s+other\s+day|other\s+than\s+that)\b", " ", t)
    t = re.sub(r"\b(?:the\s+)?" + _PRE + r"(?:-hand)?\s+" + _BINW + r"\s+(?:(?:will|wo|is|does|should|must|shall|can|ca|would|need|needs)\s+not\s+"
               r"(?:get\s+|be\s+|being\s+|to\s+be\s+)?(?:used|needed|filled|touched|involved|required|in\s+use|in\s+play)|"
               r"(?:stays?|remains?|is|will\s+be|will\s+stay|will\s+remain|should\s+stay|is\s+left|gets?\s+left)\s+"
               r"(?:completely\s+|totally\s+|entirely\s+)?(?:empty|unused|vacant|idle))\b(?!\s+(?:for|by|with))(?:\s+at\s+all)?", " , ", t)
    # words that look like positions but are not
    t = re.sub(r"\b(?:all\s*right|alright|right\s+now|right\s+away|right\s+here|right\s+there|"
               r"that's\s+right|thats\s+right|that\s+is\s+right|is\s+that\s+right|(?:get|got|do|did|doing)\s+(?:it|this|that|"
               r"them|things)\s+right|right\s+order|right\s+way|right\s+thing|right\s+after|right\s+before|right\s+in\s+front|"
               r"right\s+at|right\s+on\s+time|right\s+up|straight\s+away|right\s+where)\b", " ", t)
    t = re.sub(r"\b(?:as|like|what|how|the\s+way)\s+(?:you\s+|we\s+)?(?:always|normally|usually|typically)(?:\s+(?:do|does|did|sort|would))?\b",
               " usual ", t)
    t = re.sub(r"(^\s*|[.,;!?:\n]\s*)right\s+(?:then|so)\b", r"\1 ", t)
    t = re.sub(r"((?:^|[,.;:!?\n]|\b(?:" + _PRE + "|" + _BINW + r"|ok|okay|yeah|yes|clear|understood|it|that|there|goes|go))\s*)right\s*\?",
               r"\1 ? ", t)
    t = re.sub(r"(^[\s\-*>]*|[.!?\n]\s*)((?:ok(?:ay)?|so|well|alright)\s*,?\s*)?right\s*[,.!]", r"\1 ", t)
    t = re.sub(r"\b(?:left|right)\s*(?:and|or|/|&)\s*(?:left|right)\s+(?:are|is|as|taken|being|both|here|always|means?|refers?|"
               r"defined|judged|measured|determined|from|relative|per|according|should|must|will)\b", " ", t)
    t = re.sub(r"\bright\s+(?=beside|next\s+to|alongside|between|into|inside|in\s+the\s+(?:middle|center|centre))", " ", t)
    t = re.sub(r"\b(?:keep|keeping|leave|leaving)\s+(?:the\s+)?(?:" + _PRE + r")(?:-hand)?\s+" + _BINW +
               r"\s+(?:completely\s+|totally\s+|entirely\s+)?(?:empty|free|clear|unused|vacant|untouched|alone)\b(?!\s+(?:of|from))", " , ", t)
    t = re.sub(r"\b(?:does\s+not|doesnt|do\s+not|dont|it\s+does\s+not|not)\s+matter\s+(?:if|whether|what|which)\b|"
               r"\bbe\s+(?:it|they|that)\s+(?=an?\b|cubes?|cylinders?|prisms?|triang)", " regardless ", t)
    t = re.sub(r"(?<!to )(?<!on )(?<!at )\b(its|their|own|respective|matching|corresponding|designated|assigned|appropriate|proper|correct)"
               r"\s+right\b", r"\1 ", t)
    t = re.sub(r"\b(?:left|right)[\s-]+(?:hand|arm|gripper|manipulator|camera|wrist|finger|fingers|shoulder|elbow)s?\b"
               r"(?!\s*-?\s*(?:side|extreme|end|edge|corner|" + _BINW + r"))", " ", t)
    t = re.sub(r"\bleft\s+((?:the\s+|all\s+|all\s+the\s+|every\s+|any\s+)?" + _CRE + r"(?:\s+\w+)?)\s+(?=on\s+the\s+" + _COUNTER +
               r"|alone\b|untouched|behind|where\s+(?:they|it)|out\b|unsorted|in\s+place)", r" xleave \1 ", t)
    t = re.sub(r"\b(?:kept|left|keep|leave|stay|stays|remain|remains)\s+(?:in|at)\s+(?:their|its)\s+(?:original|current|initial|starting|"
               r"present)\s+(?:positions?|spots?|locations?)\b", " xkeep ", t)
    # "left" as in leave / remain
    t = re.sub(r"\b(?:whatever|what|whichever|anything|everything|all|ones|those|objects|items|pieces|things)\s+"
               r"(?:that\s+)?(?:is|are|'s|s|remains?)\s+left\b(?!\s+(?:on|alone|behind|untouched|where|in\s+place))", " xrest ", t)
    t = re.sub(r"\b(?:only|last)\s+(?:" + _BINW + r"\s+)?(?:that\s+is\s+|that's\s+)?left\b", " xrest ", t)
    t = re.sub(r"\b(?:whatever|whichever|what|which|the|any)\s+(?:only\s+|one\s+|last\s+|single\s+)?" + _BINW +
               r"\s+(?:that\s+)?(?:is|'s|remains|are|was|will\s+be|still)\s+(?:still\s+)?(?:left|free|empty|unused|open|vacant)\b", " xrest ", t)
    t = re.sub(r"\b(?:free|empty|unused|vacant|spare|unclaimed|unassigned)\s+" + _BINW + r"\b", " xrest ", t)
    t = re.sub(r"\bleft\s+(?:behind|unsorted|out|lying|sitting|(?:right\s+)?on\s+the\s+" + _COUNTER +
               r"|on\s+(?:the|that)\s+\w+\s+" + _COUNTER + ")", " xleave ", t)
    t = re.sub(r"\bleft\s+(?:exactly\s+|just\s+|completely\s+|entirely\s+|totally\s+)?(?:alone|untouched|as\s+is|as\s+they\s+are|"
               r"as\s+(?=it|they)|where\s+(?:they|it)(?:\s+(?:are|is|lie|lies|sit|sits))?|in\s+place|there|unchanged|unaffected)",
               " xkeep ", t)
    t = re.sub(r"\b(?:be|been|being|are|is|get|gets|were|was)\s+left\s*(?=[.,;!?\n]|$|\s+and\b|\s+but\b)",
               " xkeep ", t)
    t = re.sub(r"\b(?:on|upon)\s+(?:the|that|your)\s+(?:\w+\s+)?" + _COUNTER + r"\b|"
               r"\bout\s+of\s+(?:the|all|any|every)\s+" + _BINW + r"\b", " xcounter ", t)
    t = re.sub(r"\b(?:keep|keeping|bear|bearing|have|having)\s+(?:it\s+|this\s+|that\s+)?in\s+mind\b", " ", t)
    t = re.sub(r"(?<!by )(?<!to )(?<!on )(?<!of )(?<!from )(?<!know )(?<!see )\bwhere\s+(?:they|it)\s+(?:are|is|lie|lies|sit|sits|stand|stands|were|was)\b|\bin\s+place\b|\bput\b(?=\s*[.,;!?\n])",
               " xkeep ", t)
    t = re.sub(r"\bremains?\s+(?=xcounter|xkeep|unchanged|untouched|as\s+is|put\b|there\b|unsorted|alone)", " xkeep ", t)
    t = re.sub(r"\b(?:stays?|remains?|kept|keeps?|is|are)\s+the\s+same\b(?!\s+" + _BINW + r")", " unchanged ", t)
    # "shift every colour one bin to the left (wrapping around)"
    t = re.sub(r"\b(?:one|1|a|a\s+single)\s+(?:" + _BINW + r"|position|place|slot|step|spot|space)\s+(?:more\s+|over\s+|further\s+|"
               r"farther\s+|along\s+)?(?:to\s+|toward\s+|towards\s+)?(?:the\s+robot's\s+|the\s+|its\s+|their\s+|your\s+)?"
               r"(left|right)(?:ward|wards)?\b(?!\s+of\s+(?:the\s+)?(?:one|bin|box|container|crate|basket|tray|" + _CRE + r")\b)(?:\s+of\s+(?:its|their)\s+\w+\s+\w+)?",
               lambda m: f" xshift{m.group(1)[0]} ", t)
    t = re.sub(r"\b(?:the\s+)?" + _BINW + r"\s+(?:immediately\s+|directly\s+|just\s+|right\s+)?(?:to\s+the\s+|on\s+the\s+)?(left|right)\s+of\s+"
               r"(?:its|their)\s+(?:usual|normal|default|regular|own|standard)\s+\w+", lambda m: f" xshift{m.group(1)[0]} ", t)
    t = re.sub(r"\b(swap\w*|switch\w*|interchang\w*|exchang\w*|flip\w*|trad\w*)\s+(?:the\s+)?(?:two\s+)?(?:outer|outside|end|"
               r"outermost|edge|side)\s+(?:two\s+)?\w+", " swap left right ", t)
    t = re.sub(r"\b(?:ignor\w+|regardless\s+of|irrespective\s+of|no\s+matter|disregard\w*|never\s+mind|whatever|any|of\s+any)\s+"
               r"(?:the\s+|its\s+|their\s+|what\s+|which\s+)?(?:colou?rs?\s+(?:or|and|,)\s+(?:the\s+|its\s+)?shapes?|"
               r"shapes?\s+(?:or|and|,)\s+(?:the\s+|its\s+)?colou?rs?)\b(?:\s+(?:it|they)\s+(?:has|have|is|are))?", " ", t)
    t = re.sub(r"\b(?:every|any|all|each)\s+(?:other\s+)?(?:objects?|items?|things?|pieces?|one)\s+(?:not\s+matching|that\s+(?:does|do)\s+not\s+"
               r"match|which\s+(?:does|do)\s+not\s+match|not\s+(?:listed|mentioned|named|covered|described)|outside)\b[^.,;]*?"
               r"(?=\s+(?:shall|should|must|goes|go|is|are|will|gets?|belongs?)\b)", " xrest ", t)
    t = re.sub(r"\b(?:every|any|all|each)\s+(?:other\s+)?(?:single\s+)?(?:objects?|items?|things?|pieces?|one|ones)\s+(?:that|which)\s+"
               r"(?:is|are|does|do)\s+not\b", " xrest not ", t)
    t = re.sub(r"\b(?:do\s+not|dont|never|no\s+need\s+to)\s+(?:worry|care|bother|think|fuss|mind|fret)\s+(?:about\s+)?(?:the\s+|its\s+|their\s+)?"
               r"shapes?\b|\bshapes?\s+(?:really|truly|actually|simply|just|honestly)\s+(?:does\s+not|do\s+not|doesnt)\s+matter", " xcoloronly ", t)
    t = re.sub(r"\b(?:ignor\w+|regardless\s+of|irrespective\s+of|no\s+matter|disregard\w*|never\s+mind|whatever|any|not\s+by|"
               r"not\s+on|not)\s+(?:the\s+|its\s+|their\s+|what\s+|which\s+)?shapes?\b|\bshapes?\s+(?:does\s+not|do\s+not|is\s+not|"
               r"are\s+not|doesnt|isnt)\s+(?:matter|important|relevant|count|play)|\bshapes?\s+(?:is|are)\s+irrelevant|"
               r"\bby\s+colou?rs?\s+(?:only|alone)|\bcolou?rs?\s+(?:only|alone)\b", " xcoloronly ", t)
    t = re.sub(r"\b(?:ignor\w+|regardless\s+of|irrespective\s+of|no\s+matter|disregard\w*|never\s+mind|whatever|any|not\s+by|"
               r"not\s+on)\s+(?:the\s+|its\s+|their\s+|what\s+|which\s+)?colou?rs?\b|\bcolou?rs?\s+(?:does\s+not|do\s+not|is\s+not|"
               r"are\s+not|doesnt|isnt)\s+(?:matter|important|relevant|count|play)|\bcolou?rs?\s+(?:is|are)\s+irrelevant|"
               r"\bby\s+shapes?\s+(?:only|alone)|\bshapes?\s+(?:only|alone)\b", " xshapeonly ", t)
    t = re.sub(r"\b((?:every|all|each|any)\s+(?:the\s+|single\s+)?(?:objects?|items?|things?|pieces?|one)|everything|anything)\s*,\s*"
               r"(xshapeonly|xcoloronly)\s*,", r"\1 \2 ", t)
    t = re.sub(r"\b(" + _CRE + r")\s+(?:things?|objects?|items?|ones?|pieces?|stuff)\s+(?:that|which)\s+(?:is|are)\s+not\s+(?:an?\s+)?"
               r"((?:triangular\s+)?(?:cubes?|cylinders?|prisms?|triangles?|wedges?|blocks?))\b", r"\1 except \1 \2 ", t)
    t = re.sub(r"\b(?:round|circular)\s+(ones?|objects?|pieces?|items?|things?|shapes?|blocks?|parts?)\b", r" cylinder ", t)
    t = re.sub(r"\b(?:square|boxy|box-like)\s+(ones?|objects?|pieces?|items?|things?|shapes?|blocks?|parts?)\b", r" cube ", t)
    t = re.sub(r"\b(?:three-?\s?sided|triangular|pointy)\s+(ones?|objects?|pieces?|items?|things?|shapes?|blocks?|parts?)\b", r" prism ", t)
    t = re.sub(r"\b(?:not|never)\s+(?:be\s+|to\s+be\s+)?(?:put|placed?|go|goes|going|binned|sorted|dropped)\s+(?:\w+\s+){0,3}?"
               r"(?:anywhere|in(?:to)?\s+any\s+" + _BINW + r"|in(?:to)?\s+a\s+" + _BINW + r"|in(?:to)?\s+any\s+of\s+the\s+" + _BINW + r")\b(?!\s+(?:but|except|other|else))",
               " xleave ", t)
    if re.search(r"\b(?:keep|keeping|keeps)\s+(?:\w+\s+){1,4}?from\s+(?:the\s+)?" + _PRE + r"\b", t):
        t = re.sub(r"\b(?:keep|keeping|keeps)\s+(?=(?:\w+\s+){1,4}?from\s+(?:the\s+)?" + _PRE + r"\b)", " ", t)
        t = re.sub(r"\bfrom\s+(?=(?:the\s+)?" + _PRE + r"\b)", " xnot ", t)
    t = re.sub(r"\b(?:skips?|skipping|bypass(?:es|ing)?)\s+(?=(?:both\s+)?(?:the\s+)?" + _PRE + r"(?:\s+" + _BINW + r")?\s*(?:,|and|or|&)\s+(?:the\s+)?"
               + _PRE + r"\b)", " not in ", t)
    t = re.sub(r"\b(?:out\s+of|away\s+from|free\s+of|clear\s+of|free\s+from|off\s+limits\s+(?:to|for)|barred\s+from|"
               r"excluded\s+from|banned\s+from|kept\s+from)\b", " xnot ", t)
    t = re.sub(r"(?<!not )(?<!never )(?<!dont )(?<!cannot )(?<!to )(?<!of )(?<!beside )(?<!after )(?<!before )(?<!unless )\b((?:send|put|place|drop|move|sort|toss|deposit|route|direct|set|pop|stick|chuck|bin|take|bring|deliver)"
               r"(?:ing)?\s+(?:them|it|those|these)\b|(?:sending|putting|placing|dropping|moving|routing|directing|landing)\b|"
               r"(?:they|it|those|these)\s+(?:go|goes|belong|belongs|should|must|will|land|lands|end|ends|have|has|need|needs|"
               r"get|gets|are|is)\b|(?:instead|hence|thus|therefore|always|ergo)\b)", r" , \1", t)
    t = re.sub(r"\b(?:both|either|the\s+two|two|any|each)\s+(?:of\s+the\s+)?(?:end|outer|outside|edge|side|outermost)\s+" + _BINW +
               r"\b|\b(?:either|both|the\s+two)\s+(?:ends?|sides?|edges?)\b|\bthe\s+(?:outer|outside|end|edge|outermost)\s+(?:two\s+)?"
               + _BINW[:-1] + r"|ends)\b", " xends ", t)
    t = re.sub(r"\b(?:the\s+)?" + _CRE + r"[-\s](?:objects?|items?|things?|colou?r|coloured|colored)[-\s]rule\b", " ", t)
    t = re.sub(r"\b(overrid\w+|exceptions?\s+to|despite|unlike|contrary\s+to|trumps?|beats?|supersed\w+|break\w*)\s+(?:the\s+)?"
               r"(?:general\s+|usual\s+|normal\s+)?" + _CRE + r"(?:[-\s]\w+)?\s+rule\b", r"\1 ", t)
    t = re.sub(r"\bnon[-\s]?(" + _CRE + r")\b", r" everything except \1 ", t)
    noun_ = r"(?:\s+(?:ones?|objects?|items?|things?|pieces?|blocks?|stuff|cubes?|cylinders?|prisms?))?"
    t = re.sub(r"\b(" + _CRE + r"|cubes?|cylinders?|prisms?)" + noun_ + r"\s*,?\s+(?:do\s+|does\s+|go\s+there\s+|goes\s+there\s+|go\s+|goes\s+)?"
               r"(?:(?:too|as\s+well)\b|(?:also|likewise|ditto)\b(?=\s*[.,;!?]|\s*$|\s+and\b))", r"\1 xditto ", t)
    t = re.sub(r"\b(?:so\s+(?:is|are|do|does|will|should|must)|(?:the\s+)?same\s+(?:goes|applies|holds)\s+(?:for|to)|same\s+for|"
               r"ditto\s+for|as\s+(?:do|does|are|is))\s+(?=(?:the\s+|all\s+|every\s+|any\s+|an?\s+)?(?:" + _CRE +
               r"|cubes?|cylinders?|prisms?|triangular)\b)", " xditto ", t)
    t = re.sub(r"\bno\s+(?:\w+\s+){0,2}?(?:rule|bin|box|destination|instruction|placement|assignment|target|home)\s+"
               r"(?:is\s+|are\s+|has\s+been\s+|was\s+|exists?\s+)?(?:given\s+|assigned\s+|specified\s+|defined\s+|provided\s+|stated\s+|set\s+)?"
               r"(?:for|to)\b", " xleave ", t)
    shp = r"(?:triangular\s+)?(?:cubes?|cylinders?|prisms?|triangles?|wedges?|blocks?|pyramids?)"
    t = re.sub(r"\b(?:every|all|each|any)\s+(?:the\s+|single\s+)?" + shp + r"\s*,\s*" + shp + r"\s*,?\s*(?:and|or)\s+" + shp + r"\b", " everything ", t)
    def _three(m):
        kinds_ = {"cub": 0, "blo": 0, "cyl": 1, "pri": 2, "tri": 2, "wed": 2, "pyr": 2}
        got_ = {kinds_.get(re.sub(r"^triangular\s+", "", m.group(i))[:3]) for i in (2, 3, 4)}
        return f" {m.group(1)} objects " if got_ == {0, 1, 2} else m.group(0)
    t = re.sub(r"\b(" + _CRE + r")\s+(" + shp + r")\s*,\s*(" + shp + r")\s*,?\s*(?:and|or)\s+(" + shp + r")\b", _three, t)
    t = re.sub(r"\b(" + _CRE + r")\s+(?:and|&)\s+(" + _CRE + r")\s+((?:triangular\s+)?(?:cubes?|cylinders?|prisms?|triangles?|wedges?))\b",
               r"\1 \3 and \2 \3", t)                 # "red and blue cubes": the shape belongs to both colours
    t = re.sub(r"\bwith\s+(?=(?:the\s+|all\s+)?" + _CRE + r"\b(?:\s+\w+)?\s+(?:being|going|headed|landing|placed|put|sent|deposited)\b)", " , ", t)
    t = re.sub(r"\b(everything|all|anything)-(except|but)-", r"\1 \2 ", t)
    t = _ordinals(t)
    for a, b in _PHRASES:
        t = re.sub(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", b, t)
    t = _positions(t)
    t = _relations(t)
    def _ord_typo(m):
        a, b = _side(m.group(1)), _side(m.group(2))
        return (" xorderlr " if a == "left" else " xorderrl ") if {a, b} == {"left", "right"} else m.group(0)
    t = re.sub(r"\b(?:from\s+)?([a-z]{4,6})\s+to\s+([a-z]{4,6})\b(?=\s*(?:,|:|in\s+order|in\s+respectively|respectively|order))", _ord_typo, t)
    t = re.sub(r"\b(?:from\s+)?left\s+to\s+right\b", " xorderlr ", t)
    t = re.sub(r"\b(?:from\s+)?right\s+to\s+left\b", " xorderrl ", t)
    t = re.sub(r"\bno\s*,(?=\s*(?:the\s+|it\s+|in\s+|to\s+|on\s+)*(?:" + _PRE + "|" + _CRE + r")\b)", " xfix ", t)
    t = re.sub(_FIXRE, " xfix ", t)
    t = re.sub(r"xfix\s*,?\s*(?:no|nope|not\s+that)\b\s*,?", " xfix ", t)
    # a line break ends a statement unless the line ends in ':' or ','
    t = re.sub(r"([:,])\s*\n+", r"\1 ", t)
    t = re.sub(r"\n+", " . ", t)
    return t


def _tokens(text: str):
    """[(kind, value)] with kind in C H P NEG EXC ALL SWAP REV USUAL RESP REST LEAVE ACT ONLY FIX X ORDER REL
    S (strong separator) W (weak separator)."""
    t = _normalise(text)
    raw = re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*|[.;:!?]|,", t)
    out = []
    for i, w in enumerate(raw):
        w = w.strip("'")
        if w.endswith("'s") and w[:-2] in _LOOKUP:
            w = w[:-2]
        if w == ":" and len(out) >= 2 and out[-1][0] in ("P", "C") and out[-2][0] in ("P", "C") and out[-2][0] != out[-1][0]:
            out.insert(len(out) - 1, ("W", ","))       # "left: blue center: red" -> the header starts a new clause
        if w in _CONTRAST and w != "but":
            out.append(("W", "contrast"))
        elif w == ":":
            out.append(("W", ":"))                    # clause break; leftover bins carry over ("Left bin: yellow")
        elif w in (".", ";", "!", "?"):
            out.append(("S", w))
        elif w == ",":
            out.append(("W", w))
        elif w == "xfix":
            out.append(("FIX", w))
        elif w == "xarrow":
            out.append(("W", "arrow"))
        elif w in ("xorderlr", "xorderrl"):
            out.append(("ORDER", w[-2:]))
        elif w.startswith("xdef") and w[4:] in COLORS:
            out.append(("D", w[4:]))                  # that colour's bin under the default rule
        elif w == "xcounter":
            out.append(("CTR", w))
        elif w == "xditto":
            out.append(("XD", w))
        elif w == "which" and out and out[-1][0] == "W":
            out.append(("WH", w))
        elif w == "xopp":
            out.append(("P", ("opp",)))
        elif w in ("xshiftl", "xshiftr"):
            out.append(("SHIFT", -1 if w[-1] == "l" else 1))
        elif w in _KEEP:
            out.append(("KEEP", w))
        elif w.startswith("xrel") and w[5:] in COLORS:
            out.append(("P", ("rel", w[4], w[5:])))
        elif w.startswith("xsame") and w[5:] in COLORS:
            out.append(("P", ("rel", "s", w[5:])))
        elif w == "xsamebin":
            out.append(("P", ("last",)))
        elif w in ("xadjl", "xadjr", "xadjn"):
            out.append(("P", ("adj", w[-1])))
        elif w == "xlastbin":
            out.append(("P", ("lastbin",)))
        elif w == "xends":                             # "both end bins": only usable as an exclusion
            if _neg_context(out):
                out += [("P", "left"), ("P", "right")]
        elif (w in ("and", "or", "nor", "&", ",") and out and out[-1][0] == "P" and _next_is_pos(raw, i)
              and (_neg_context(out) or _plural_bin_after(raw, i))):
            continue                                   # "not in the left or center bin": one list of bins
        elif w in ("xcoloronly", "xshapeonly"):
            out.append(("MODE", w[1:]))
        elif w in _ASIDE:
            out.append(("ASIDE", w))
        elif w in _BLOCK:
            out.append(("HB", "cube"))
        elif w in _LOOKUP:
            out.append(_LOOKUP[w])
        elif w.strip("-") in _LOOKUP:
            out.append(_LOOKUP[w.strip("-")])
        elif w in _NEG:
            out.append(("NEG", w))
        elif w == "but":
            near = raw[max(0, i - 4):i]
            out.append(("EXC", w) if any((x in _ALL and x not in _ALL_NOUN) or x == "none" for x in near) else ("W", "contrast"))
        elif w in _EXCEPT:
            out.append(("EXC", w))
        elif w in _SWAP:
            out.append(("SWAP", w))
        elif w in _REVERSE:
            out.append(("REV", w))
        elif w in _USUAL:
            out.append(("USUAL", w))
        elif w == "respectively" or w == "vastaavasti":
            out.append(("RESP", w))
        elif w in _REST:
            out.append(("REST", w))
        elif w in _LEAVE:
            out.append(("LEAVE", w))
        elif w in _ACT:
            out.append(("ACT", w))
        elif w in _ONLY:
            if any(_LOOKUP.get(x, ("",))[0] == "C" for x in raw[max(0, i - 2):i] + raw[i + 1:i + 4]):
                out.append(("ONLY", w))
        elif w in _TOO:
            out.append(("X", w))
        elif w in _WEAK:
            out.append(("W", w))
        elif w in _ALL:
            if not (w in _ALL_NOUN and out and out[-1][0] in ("C", "H", "HB")):
                out.append(("ALL", w))
        else:
            hit = None
            for stem, tok in _STEMS:
                if w.startswith(stem):
                    hit = [tok]
                    break
            if hit is None and "-" in w:              # compound like "red-objects", "left-bin", "cherry-red"
                hit = list(dict.fromkeys(_LOOKUP[p] for p in w.split("-") if p in _LOOKUP)) or None
            if hit is None:
                f = _fuzzy(w)
                hit = [f] if f else _segment(w)
            for tok in hit or ():
                out.append(tok)
    return out


_FILL = {"the", "in", "into", "to", "on", "from", "of", "a", "at", "either", "even", "that"}


def _next_is_pos(raw, i):
    j = i + 1
    while j < len(raw) and raw[j] in _FILL:
        j += 1
    return j < len(raw) and _LOOKUP.get(raw[j], ("",))[0] == "P"


def _plural_bin_after(raw, i):
    j = i + 1
    while j < len(raw) and raw[j] in _FILL:
        j += 1
    return j + 1 < len(raw) and re.fullmatch(r"bins|boxes|containers|crates|baskets|trays|buckets|tubs|slots|ones", raw[j + 1]) is not None


def _neg_context(out):
    for k, _ in reversed(out):
        if k in ("S", "C"):
            return False
        if k == "NEG":
            return True
    return False


def _split_neg(cl):
    """Scope of a negation inside one clause: 'yellow goes left not right' -> [yellow left] [not right];
    'yellow never right blue center' -> [yellow never right] [blue center]."""
    ks = [k for k, _ in cl]
    if "NEG" not in ks or "EXC" in ks or "REST" in ks or "RESP" in ks:
        return [cl]
    i = ks.index("NEG")
    out, post = [], cl
    if "C" in ks[:i] and "P" in ks[:i]:
        out.append(cl[:i])
        post = cl[i:]
    seen_neg = seen_p = False
    for j, (k, _) in enumerate(post):
        if k == "NEG":
            seen_neg = True
        elif k == "P" and seen_neg:
            seen_p = True
        elif k == "C" and seen_neg and seen_p:
            return out + [post[:j]] + _split_neg(post[j:])
    return out + [post]


def _attach_shapes(toks):
    """Merge SHAPE tokens with the colour of their noun phrase: 'yellow cubes and cylinders' -> (yellow, cube),
    (yellow, cylinder). Returns tokens with kind C whose value is a colour, a shape or a (colour, shape) tuple."""
    out = []
    np_color = None        # colour of the current noun phrase
    for k, v in toks:
        if k == "C":
            np_color = v
            # postfix colour: "cubes that are red" -> recolour the directly preceding run of bare shapes
            j = len(out) - 1
            touched = False
            while j >= 0 and (out[j][0] == "W" or (out[j][0] == "C" and out[j][1] in SHAPES)):
                if out[j][0] == "C":
                    out[j] = ("C", (v, out[j][1]))
                    touched = True
                j -= 1
            if touched:
                continue
            out.append((k, v))
        elif k == "H":
            if np_color is not None:
                if out and out[-1] == ("C", np_color):
                    out[-1] = ("C", (np_color, v))
                    j = len(out) - 2                   # "blue or yellow cubes": every colour of the list
                    while j >= 0 and out[j][0] == "C" and out[j][1] in COLORS:
                        out[j] = ("C", (out[j][1], v))
                        j -= 1
                else:
                    out.append(("C", (np_color, v)))
            else:
                out.append(("C", v))
        else:
            if k in ("P", "S", "EXC", "REST", "ALL", "NEG", "LEAVE", "FIX"):
                np_color = None
            out.append((k, v))
    return out


def _runs(items):
    """[(kind, value)] of C/P -> runs [(kind, [values])]."""
    runs = []
    for k, v in items:
        if runs and runs[-1][0] == k:
            runs[-1][1].append(v)
        else:
            runs.append((k, [v]))
    return runs


def _pair_runs(items, cross=False):
    """Pair colour runs with position runs inside one clause. Returns (pairs, leftover colours, leftover positions)."""
    runs = _runs(items)
    pairs, i = [], 0
    left_c, left_p = [], []
    while i < len(runs):
        k, vals = runs[i]
        if i + 1 < len(runs):
            k2, vals2 = runs[i + 1]
            cs, ps = (vals, vals2) if k == "C" else (vals2, vals)
            if len(cs) == len(ps) and not (cross and len(cs) > 1):
                pairs += list(zip(cs, ps))
            elif cross:                               # "red and blue never go left or right": every combination
                pairs += [(c, p) for c in cs for p in ps]
            elif len(ps) == 1:
                pairs += [(c, ps[0]) for c in cs]
            elif len(cs) == 1:
                # "red goes left or center": weak evidence for each
                pairs += [(cs[0], p) for p in ps]
            else:
                n = min(len(cs), len(ps))
                pairs += list(zip(cs[:n], ps[:n]))
            i += 2
        else:
            (left_c if k == "C" else left_p).extend(vals)
            i += 1
    return pairs, left_c, left_p


class _Ev:
    """Evidence collected from the token stream. Items carry (epoch, order): a later self-correction wins."""

    def __init__(self):
        self.pos, self.neg, self.rel, self.leave = [], [], [], []   # (key, bin|rel, epoch, n)
        self.rest_bins, self.all_bins = [], []
        self.rest_leave = self.rest_rembin = self.only = self.usual = self.reverse = False
        self.rembin = []
        self.swaps, self.notes, self.cc = [], [], []
        self.shift = 0
        self.mentioned = set()
        self.kept = set()
        self.epoch = self.n = 0
        self.last_group = []

    def add(self, pairs, neg=False):
        grp = []
        for c, p in pairs:
            self.n += 1
            if isinstance(p, tuple):
                if p[0] == "last":
                    if not self.last_group:
                        continue
                    p = self.last_group[-1][1]
                elif p[0] == "lastbin":
                    free = [b for b in BINS if b not in {x[1] for x in self.pos} | {g[1] for g in grp}]
                    p = free[0] if len(free) == 1 else "right"
                elif p[0] == "opp":
                    prev = (grp or self.last_group or [(None, None)])[-1][1]
                    if prev not in ("left", "right"):
                        continue
                    p = "right" if prev == "left" else "left"
                elif p[0] == "adj":
                    prev = (grp or self.last_group or [(None, None)])[-1][1]
                    if prev not in BINS:
                        continue
                    used = {x[1] for x in self.pos} | {g[1] for g in grp}
                    free = [b for b in BINS if b not in used]
                    if p[1] == "n":
                        if prev != "center":
                            p = "center"
                        elif len(free) == 1:
                            p = free[0]
                        else:
                            continue
                    else:                             # "left of it": the bin just named, else an earlier anchor
                        cands = [prev] + [x[1] for x in reversed(self.pos)]
                        step = -1 if p[1] == "l" else 1
                        hit = [BINS[BINS.index(b) + step] for b in cands if 0 <= BINS.index(b) + step <= 2]
                        hit = [b for b in hit if b in free] or hit[:1]
                        if not hit:
                            continue
                        p = hit[0]
                else:
                    if not neg and c != p[2]:
                        self.rel.append((c, p, self.epoch, self.n))
                    continue
            (self.neg if neg else self.pos).append((c, p, self.epoch, self.n))
            if not neg:
                grp.append((c, p))
        if grp:
            self.last_group = grp

    def current(self, items, key):
        """Items about `key` from the latest epoch in which the key was mentioned at all."""
        e = max([x[2] for lst in (self.pos, self.neg, self.leave) for x in lst if x[0] == key], default=0)
        return [x for x in items if x[0] == key and x[2] == e]


def _collect(toks, universe):
    ev = _Ev()
    sentences, cur = [], []
    for tk in toks:
        if tk[0] == "S":
            if cur:
                sentences.append((cur, tk[1]))
            cur = []
        else:
            cur.append(tk)
    if cur:
        sentences.append((cur, "."))
    q_p = prev_three = None
    lists = zipped = None      # a bare list of three colours or three bins in the previous sentence / the colours last zipped

    pending_order = None
    fix_carry = False
    pending_keys = []          # colours whose last statement was only negative / had no bin: "put them left" binds here
    color_mode = tuple(universe) == COLORS
    ev.mentioned = {v[0] if isinstance(v, tuple) else v for k, v in toks if k in ("C", "D")}
    default_rel = color_mode and any(k in ("USUAL", "SWAP", "REV", "D", "SHIFT") for k, _ in toks)
    for sent, term in sentences:
        start_p, q_p = q_p, None
        kinds = [k for k, _ in sent]
        cs = [v for k, v in sent if k == "C"]
        ps = [v for k, v in sent if k == "P"]
        three, prev_three = prev_three, (list(dict.fromkeys(cs)) if len(set(cs)) == 3 and not ps else None)
        cds = [v for k, v in sent if k in ("C", "D")]
        bare = not (set(kinds) - {"C", "P", "W", "RESP", "FIX", "X"}) and not any(isinstance(p, tuple) for p in ps)
        prev_list, lists = lists, None
        if bare and not cs and len(ps) == 3 and len(set(ps)) == 3:
            if prev_list and prev_list[0] == "C" and ("RESP" in kinds or prev_list[2]):
                ev.add(list(zip(prev_list[1], ps)))
                ev.notes.append("lists")
                zipped = prev_list[1]
                continue
            if zipped and "FIX" in kinds:              # "Sorry, I meant left, center, right."
                ev.epoch += 1
                ev.add(list(zip(zipped, ps)))
                ev.notes.append("correction")
                continue
            lists = ("P", list(ps), "RESP" in kinds)
        elif bare and not ps and len(cs) == 3 and len(set(cs)) == 3:
            if prev_list and prev_list[0] == "P" and ("RESP" in kinds or prev_list[2]):
                ev.add(list(zip(cs, prev_list[1])))
                ev.notes.append("lists")
                zipped = list(cs)
                continue
            lists = ("C", list(cs), "RESP" in kinds)
        if "ONLY" in kinds:
            ev.only = True
        if "USUAL" in kinds or "D" in kinds:
            ev.usual = True
        if "SHIFT" in kinds and ("ALL" in kinds or default_rel):
            ev.shift = [v for k, v in sent if k == "SHIFT"][0]
            ev.notes.append("shift")
        if "SWAP" in kinds and (not ps or not cs):
            idx = [i for i, k in enumerate(kinds) if k == "SWAP"]
            segs = []
            for j, i in enumerate(idx):
                seg = [v for k, v in sent[i:idx[j + 1] if j + 1 < len(idx) else len(sent)] if k in ("C", "D")]
                if len(seg) >= 2 and seg[0] != seg[1]:
                    segs.append(seg)
            before = list(dict.fromkeys(v for k, v in sent[:idx[0]] if k in ("C", "D")))
            two = list(dict.fromkeys(cds))
            if segs:
                ev.swaps += [("C", g[0], g[1]) for g in segs]
            elif len(before) >= 2:
                ev.swaps.append(("C", before[-2], before[-1]))
            elif len(two) >= 2:
                ev.swaps.append(("C", two[0], two[1]))
            elif len(ps) >= 2:
                ev.swaps.append(("P", ps[-2], ps[-1]))
            elif ("REV" in kinds or "ORDER" in kinds) and "NEG" not in kinds[:idx[0]]:      # "flip it left to right" = mirror
                ev.reverse = True
                ev.usual = ev.usual or "USUAL" in kinds
                ev.notes.append("reverse")
                continue
            ev.notes.append("swap")
            continue
        if "REV" in kinds and not ps:
            two = list(dict.fromkeys(cds))
            if len(two) == 2:                          # "reverse the bins of red and yellow" = swap those two
                ev.swaps.append(("C", two[0], two[1]))
                ev.notes.append("swap")
                continue
        if "D" in kinds:
            sent = [("P", FIXED[v]) if k == "D" else (k, v) for k, v in sent]
            kinds = [k for k, _ in sent]
            ps = [v for k, v in sent if k == "P"]
        if ("ORDER" in kinds or pending_order) and "REV" not in kinds:
            order = [v for k, v in sent if k == "ORDER"]
            order = order[0] if order else pending_order
            pending_order = None
            ucs = list(dict.fromkeys(cs))
            if len(ucs) == 3 and not [p for p in ps if not isinstance(p, tuple)]:
                bins = BINS if order == "lr" else BINS[::-1]
                ev.add(list(zip(ucs, bins)))
                ev.notes.append("order")
                continue
            if not cs and "ORDER" in kinds:
                if three and not ps:                   # "red, yellow and blue. From the left, in that same order."
                    ev.add(list(zip(three, BINS if order == "lr" else BINS[::-1])))
                    ev.notes.append("order")
                else:
                    pending_order = order
                continue
        if "REV" in kinds:
            ev.reverse = True
            ev.notes.append("reverse")
        post_fix = None
        if "FIX" in kinds:
            jf = max(i for i, k in enumerate(kinds) if k == "FIX")
            fc = list(dict.fromkeys(v for k, v in sent[:jf] if k == "C"))
            fp = [v for k, v in sent[jf:] if k == "P" and not isinstance(v, tuple)]
            if len(fc) == len(fp) >= 2 and not (set(kinds[jf:]) & {"C", "NEG", "EXC", "REST", "LEAVE", "KEEP", "ALL"}):
                post_fix = list(zip(fc, fp))
                sent = sent[:jf]
                kinds = [k for k, _ in sent]
                cs = [v for k, v in sent if k == "C"]
                ps = [v for k, v in sent if k == "P"]
        if "RESP" in kinds:
            ir = kinds.index("RESP")
            hc = [v for k, v in sent[:ir] if k == "C"]
            hp = [v for k, v in sent[:ir] if k == "P"]
            if len(hc) >= 2 and len(hc) == len(hp):
                ev.add(list(zip(hc, hp)), neg=any(k == "NEG" for k in kinds[:ir]))
                ev.notes.append("respectively")
                sent = sent[ir + 1:]
                kinds = [k for k, _ in sent]
            elif cs and len(cs) == len(ps):
                last_cp = max(i for i, k in enumerate(kinds) if k in ("C", "P"))
                ev.add(list(zip(cs, ps)), neg=any(k == "NEG" for k in kinds[:last_cp]))
                ev.notes.append("respectively")
                continue
        cp = [(k, v) for k, v in sent if k in ("C", "P", "NEG")]
        if cp and cp[0][0] == "NEG" and sum(k == "NEG" for k, _ in cp) == 1 \
                and not (set(kinds) & {"EXC", "REST", "LEAVE", "KEEP", "ALL", "SWAP", "REV", "USUAL", "FIX", "CTR"}):
            groups = []
            for k, v in cp[1:]:
                if k == "C":
                    if groups and not groups[-1][1]:
                        groups[-1][0].append(v)
                    else:
                        groups.append(([v], []))
                elif groups and not isinstance(v, tuple):
                    groups[-1][1].append(v)
            if len(groups) >= 2 and all(len(set(g[1])) == 2 and len(g[1]) == 2 for g in groups):
                ev.add([(c_, p) for cs_, ps_ in groups for c_ in cs_ for p in ps_], neg=True)
                ev.notes.append("negative-list")
                continue
        # ---- clauses (weak separators); carries live inside one sentence only
        clauses, c, fixed, sep = [], [], fix_carry, ""
        fix_carry = False
        for tk in sent:
            if tk[0] in ("W", "FIX"):
                if c:
                    has_cp = any(k_ in ("C", "P") for k_, _ in c)
                    clauses.append((c, fixed and has_cp, sep))
                    fixed, sep = fixed and not has_cp, ""
                c = []
                if tk[0] == "FIX":
                    fixed = True
                elif sep != "contrast":
                    sep = tk[1]
            else:
                c.append(tk)
        if c:
            clauses.append((c, fixed, sep))
        fix_carry = fixed and not any(k_ in ("C", "P") for k_, _ in c)      # "Wait, no. The center bin takes blue."
        clauses = [(part, fixed and n_ == 0, sep_ if n_ == 0 else "")
                   for cl_, fixed, sep_ in clauses for n_, part in enumerate(_split_neg(cl_))]
        exc_pending, exc_fresh, exc_own, exc_at = [], False, False, -9
        exc_wh = False
        last_p, carry_c, carry_p, rest_pending, p_first = None, [], ([start_p] if start_p else []), False, False
        neg_p = []
        leave_open = False
        carry_and = False
        for ci, (cl, fixed, sep) in enumerate(clauses):
            ks = [k for k, _ in cl]
            if sep == "contrast":                     # "..., while the yellow ones ..." starts a new statement
                if carry_c and p_first and last_p is not None:
                    ev.add([(c_, last_p) for c_ in carry_c])
                carry_c, carry_p = [], []
            if "EXC" in ks:                           # "everything except blue goes right blue goes center": cut after the bin
                ie = ks.index("EXC")
                jc = next((j for j in range(ie + 1, len(cl)) if ks[j] == "C"), None)
                jp = next((j for j in range(jc + 1, len(cl)) if ks[j] == "P"), None) if jc is not None else None
                if jp is not None and "C" in ks[jp + 1:]:
                    while jp + 1 < len(cl) and ks[jp + 1] == "P":
                        jp += 1
                    clauses.insert(ci + 1, (cl[jp + 1:], False, ""))
                    cl = cl[:jp + 1]
                    ks = [k for k, _ in cl]
            cp_idx = [i for i, k in enumerate(ks) if k in ("C", "P")]
            ps_cl = [v for k, v in cl if k == "P"]
            is_neg = bool(cp_idx) and any(k == "NEG" or (k == "LEAVE" and ps_cl and v in ("exclude", "skip", "omit", "ignore"))
                                          for k, v in cl[:cp_idx[-1]])
            cs_cl = [v for k, v in cl if k == "C"]
            if leave_open and cs_cl and not ps_cl and (sep in ("and", "or", "ja", "und", "et", "y", "och", "e", "en") or "CTR" in ks) \
                    and not (set(ks) - {"C", "CTR", "ALL", "X", "XD", "KEEP"}):
                for k_ in cs_cl:                       # "leave red and blue on the counter"
                    ev.n += 1
                    ev.leave.append((k_, None, ev.epoch, ev.n))
                carry_c, carry_p = [], []
                continue
            leave_open = False
            if fixed and (cs_cl or ps_cl):
                ev.epoch += 1
                ev.notes.append("correction")
                grp = ev.last_group
                if grp and ps_cl and not cs_cl and not is_neg and not carry_c:          # "red left, no wait, right"
                    ev.pos = [x for x in ev.pos if (x[0], x[1]) not in grp]
                    ev.add([(k_, ps_cl[-1]) for k_ in dict.fromkeys(g[0] for g in grp)])
                    carry_c, carry_p = [], []
                    continue
                if grp and cs_cl and not ps_cl and ci == len(clauses) - 1 and "LEAVE" not in ks:   # "left gets red, no, blue"
                    ev.pos = [x for x in ev.pos if (x[0], x[1]) not in grp]
                    ev.add([(k_, grp[-1][1]) for k_ in cs_cl])
                    carry_c, carry_p = [], []
                    continue
            # ---- leave on the counter: "leave the blue ones", "don't touch yellow", "ignore everything else"
            if "KEEP" in ks and cs_cl and not ps_cl and not (set(ks) - {"KEEP", "C", "ALL"}) and ci + 1 < len(clauses) \
                    and all(v in ("keep", "keeps", "keeping") for k, v in cl if k == "KEEP") \
                    and clauses[ci + 1][2] in ("and", "or", "ja", "und", "et", "y", "och", "e", "en") and {"C", "P"} <= {k for k, _ in clauses[ci + 1][0]} \
                    and not ({k for k, _ in clauses[ci + 1][0]} & {"NEG", "KEEP", "LEAVE", "CTR", "ACT", "EXC", "REST", "USUAL"}):
                cl = [x for x in cl if x[0] != "KEEP"]      # "keep red and yellow together in the right bin"
                ks = [k for k, _ in cl]
            lk = [i for i, k in enumerate(ks) if k in ("KEEP", "LEAVE")]
            neg_before = bool(lk) and "NEG" in ks[:lk[0]]
            keep_word = "KEEP" in ks and not neg_before and "LEAVE" not in ks and ("NEG" not in ks or "CTR" in ks)
            neg_soft = "NEG" in ks and any(k == "ACT" and v in _SOFT_ACT for k, v in cl)
            strong = "USUAL" in ks or any(k == "KEEP" and v in _KEEP_STRONG for k, v in cl)
            if color_mode and not ps_cl and "CTR" not in ks and ((keep_word and (strong or default_rel))
                                                                 or (neg_soft and default_rel)):
                # "leave yellow as usual" / (in a rule stated relative to the default) "yellow stays put"
                keys = [k_ for k_ in (cs_cl or carry_c) if k_ in FIXED]
                if "EXC" in ks:
                    ie = ks.index("EXC")
                    keys = [k_ for k_ in COLORS if k_ not in [v for k, v in cl[ie + 1:] if k == "C"]]
                if keys:
                    ev.add([(k_, FIXED[k_]) for k_ in keys])
                    ev.kept.update(keys)
                    ev.notes.append("keep-usual")
                if keys or default_rel or "ALL" in ks:      # "keep these destinations unchanged" says nothing about a colour
                    ev.usual = True
                carry_c, carry_p = [], []
                continue
            if color_mode and "USUAL" in ks and cs_cl and not ps_cl and not (set(ks) & {"EXC", "LEAVE", "REST"}) \
                    and not any(k == "NEG" and v not in ("without", "no") for k, v in cl):
                keys = [k_ for k_ in cs_cl if k_ in FIXED]      # "the usual rule governs blue"
                ev.add([(k_, FIXED[k_]) for k_ in keys])
                ev.kept.update(keys)
                ev.notes.append("keep-usual")
                carry_c, carry_p = [], []
                continue
            leave_word = bool(lk) and not neg_before and ("NEG" not in ks or "CTR" in ks or ks[lk[0]] == "LEAVE")
            neg_act = ("NEG" in ks and "ACT" in ks and "LEAVE" not in ks and "EXC" not in ks
                       and (cs_cl or carry_c or "REST" in ks or exc_pending))
            if (leave_word or neg_act) and not ps_cl:
                keys = cs_cl or carry_c
                if not keys and exc_pending and "EXC" not in ks:      # "the exception is blue prisms, which stay on the counter"
                    keys = [u for u in universe if u not in exc_pending] if ("ALL" in ks or "REST" in ks) else list(exc_pending)
                    exc_pending = []
                if not keys and pending_keys and leave_word and "CTR" in ks and not (set(ks) & {"EXC", "REST", "ALL"}):
                    keys, pending_keys = list(pending_keys), []      # "Red prisms are the exception; leave them on the counter."
                if "EXC" in ks:
                    ie = ks.index("EXC")
                    keep = [v for k, v in cl[ie + 1:] if k == "C"]
                    keys = [u for u in universe if u not in keep] if keep else keys
                    if keep and any(isinstance(u, tuple) or u not in universe for u in keep):
                        ev.rest_leave = True
                        keys = []
                if keys:
                    for k_ in keys:
                        ev.n += 1
                        ev.leave.append((k_, None, ev.epoch, ev.n))
                    ev.notes.append("leave")
                    leave_open = bool(cs_cl) and "EXC" not in ks and "CTR" not in ks
                elif "REST" in ks or ("ALL" in ks and leave_word):
                    ev.rest_leave = True
                    ev.notes.append("rest-leave")
                carry_c, carry_p = [], []
                continue
            if ("REST" in ks or rest_pending) and ps_cl and (not cs_cl or "NEG" in ks):  # "the rest go to the left bin"
                p_ = [p for p in ps_cl if not isinstance(p, tuple)]
                if p_:
                    ev.rest_bins.append(p_[-1])
                carry_c, carry_p, rest_pending = [], [], False
                continue
            if "REST" in ks and not ps_cl:
                if cs_cl and "EXC" not in ks and "NEG" not in ks:      # "blue goes in the remaining bin"
                    ev.rembin += cs_cl
                    pending_keys = list(cs_cl)
                elif not cs_cl:
                    rest_pending = True
                    ev.rest_rembin = True
                carry_c, carry_p = [], []
                continue
            if rest_pending and "NEG" in ks and cs_cl and not ps_cl:   # "..., meaning anything that is neither red nor yellow, ..."
                continue
            # "never put red anywhere except the right" = red -> right
            if "NEG" in ks and "EXC" in ks:
                ie = ks.index("EXC")
                after_c = [v for k, v in cl[ie + 1:] if k == "C"]
                after_p = [v for k, v in cl[ie + 1:] if k == "P"]
                before_c = [v for k, v in cl[:ie] if k == "C"]
                if after_p and not after_c and before_c:
                    ev.add([(c_, after_p[0]) for c_ in before_c])
                    carry_c, carry_p = [], []
                    continue
            def general_rule(excl, p):
                if all(e in universe for e in excl) and not isinstance(universe, list):
                    ev.add([(c_, p) for c_ in universe if c_ not in excl])
                else:
                    ev.rest_bins.append(p)
                ev.add([(c_, p) for c_ in excl], neg=True)

            plain_all = [p for p in ps_cl if not isinstance(p, tuple)]
            if exc_pending and "WH" in ks and not cs_cl and not ps_cl and not (set(ks) - {"WH", "X"}):
                exc_wh = True                          # "unless it is a blue cylinder, in which case, it goes ..."
                continue
            if exc_pending:
                if cs_cl and not ps_cl and not (set(ks) & {"NEG", "EXC", "LEAVE", "KEEP", "REST"}) and exc_fresh \
                        and sep in ("and", "or", "ja", "und", "et", "y", "och", "e", "en"):
                    exc_pending += cs_cl               # "with the exception of red and yellow stuff, ..."
                    continue
                if exc_fresh and ci == exc_at + 1 and cs_cl and ps_cl and not (set(ks) & {"NEG", "ACT", "REST", "ALL", "EXC"}) \
                        and ks.index("C") < ks.index("P") and not isinstance(cl[ks.index("P")][1], tuple) and sep in ("and", "or", "ja", "und", "et", "y", "och", "e", "en"):
                    ip = ks.index("P")                 # "all but the red and yellow ones go in the center bin ..."
                    exc_pending += [v for k, v in cl[:ip] if k == "C"]
                    general_rule(exc_pending, cl[ip][1])
                    pending_keys, exc_pending, exc_fresh = list(exc_pending), [], False
                    if cl[ip + 1:]:
                        clauses.insert(ci + 1, (cl[ip + 1:], False, ""))
                    carry_c, carry_p = [], []
                    continue
                exc_fresh = False
                if plain_all and not cs_cl and ("WH" in ks or exc_wh) and "NEG" not in ks:
                    exc_wh = False
                    ev.add([(c_, plain_all[-1]) for c_ in exc_pending])   # "apart from blue, which go left, ..."
                    exc_own = True
                    continue
                if plain_all and not cs_cl and "NEG" not in ks and "LEAVE" not in ks:
                    general_rule(exc_pending, plain_all[-1])   # "..., put everything in the center bin"
                    pending_keys, exc_pending = ([] if exc_own else list(exc_pending)), []
                    carry_c, carry_p = [], []
                    continue
                exc_pending = []
            if "EXC" in ks and not carry_c:           # "... , except that blue cubes go right" / "... , except blue"
                ie = ks.index("EXC")
                head = [k for k in ks[:ie] if k in ("C", "P", "REST", "NEG")]
                excl_ = [v for k, v in cl[ie + 1:] if k == "C"]
                p_aft = [v for k, v in cl[ie + 1:] if k == "P" and not isinstance(v, tuple)]
                strong = [j for j in range(ie + 1, len(cl)) if ks[j] == "ALL" and cl[j][1] not in _ALL_NOUN]
                first_c = next((j for j in range(ie + 1, len(cl)) if ks[j] == "C"), len(cl))
                strong = [j for j in strong if j > first_c]
                if excl_ and not p_aft and not [k for k in ks[:ie] if k == "P"] and "LEAVE" not in ks and "KEEP" not in ks \
                        and not (ev.all_bins or last_p) and "REST" not in ks:
                    exc_pending = list(excl_)          # the exception comes first, the rule follows
                    exc_fresh, exc_own, exc_at = True, False, ci
                    carry_c, carry_p = [], []
                    continue
                if not head and excl_ and strong and "ALL" not in ks[:ie] and "REST" not in ks:
                    after_all = [v for k, v in cl[strong[0]:] if k == "P" and not isinstance(v, tuple)]
                    before_all = [v for k, v in cl[ie + 1:strong[0]] if k == "P" and not isinstance(v, tuple)]
                    if after_all:                     # "except blue (which go right) everything goes center"
                        general_rule(excl_, after_all[-1])
                        if before_all:
                            ev.add([(c_, before_all[-1]) for c_ in excl_])
                            pending_keys = []
                        else:
                            pending_keys = list(excl_)
                        carry_c, carry_p = [], []
                        continue
                if not head and excl_ and "ALL" not in ks[:ie] and not strong and "REST" not in ks:
                    if p_aft:
                        ev.add([(c_, p_aft[-1]) for c_ in excl_], neg="NEG" in ks[ie:])
                        pending_keys = []
                    elif "LEAVE" in ks or "KEEP" in ks:
                        for k_ in excl_:
                            ev.n += 1
                            ev.leave.append((k_, None, ev.epoch, ev.n))
                    else:
                        base = ev.all_bins[-1] if ev.all_bins else last_p
                        if base:
                            ev.add([(c_, base) for c_ in excl_], neg=True)
                        pending_keys = list(excl_)
                    carry_c, carry_p = [], []
                    continue
            # EXCEPT: "everything except blue goes left" / "all objects but the red ones in the center bin"
            if "EXC" in ks:
                ie = ks.index("EXC")
                excl = [v for k, v in cl[ie + 1:] if k == "C"]
                incl = [v for k, v in cl[:ie] if k == "C"]
                p_before = [v for k, v in cl[:ie] if k == "P" and not isinstance(v, tuple)]
                p_after = [v for k, v in cl[ie + 1:] if k == "P" and not isinstance(v, tuple)]
                if excl and (p_before or p_after):
                    p_main = (p_before or p_after)[-1]
                    targets = incl or [c_ for c_ in universe if c_ not in excl]
                    general = not incl and any(isinstance(e, tuple) or e not in universe for e in excl)
                    if general:                       # "everything except the red cubes goes left"
                        ev.rest_bins.append(p_main)
                    else:
                        ev.add([(c_, p_main) for c_ in targets], neg=is_neg and not p_before)
                    ev.add([(c_, p_main) for c_ in excl], neg=True)
                    if p_before and p_after:          # "... go left except blue, which goes right"
                        ev.add([(c_, p_after[-1]) for c_ in excl])
                        pending_keys = []
                    else:
                        pending_keys = list(excl)
                    carry_c, carry_p = [], []
                    continue
            items = [(k, v) for k, v in cl if k in ("C", "P")]
            if cs_cl and not ps_cl and not carry_c and ev.last_group and "XD" in ks \
                    and not (set(ks) & {"NEG", "EXC", "LEAVE", "KEEP", "REST"}):
                ev.add([(c_, ev.last_group[-1][1]) for c_ in cs_cl])       # "yellow too", "blue as well"
                ev.notes.append("carried")
                continue
            if carry_c and carry_and and p_first and last_p is not None and items and items[0][0] == "C" and ps_cl and not is_neg \
                    and sep == "," and not carry_p:
                ev.add([(c_, last_p) for c_ in carry_c])                   # "right bin for red and blue, yellow in the center"
                carry_c = []
            if carry_c and p_first and last_p is not None and items and items[0][0] == "P" and cs_cl and not is_neg:
                ev.add([(c_, last_p) for c_ in carry_c])                   # "center gets yellow and blue, left gets red"
                carry_c = []
            if len(cs_cl) == 2 and not ps_cl and cs_cl[0] != cs_cl[1] and all(c_ in COLORS for c_ in cs_cl) \
                    and not (set(ks) & {"NEG", "EXC", "KEEP", "LEAVE", "REST", "ALL"}):
                ev.cc.append((cs_cl[0], cs_cl[1]))     # "red takes yellow" (colour -> colour, relative to the default)
            if "ALL" in ks and not cs_cl and not carry_c and len(ps_cl) == 1 and not isinstance(ps_cl[0], tuple) \
                    and not (set(ks) & {"KEEP", "USUAL"}):
                if is_neg:                            # "nothing goes in the left bin"
                    ev.add([(c_, ps_cl[0]) for c_ in universe], neg=True)
                else:                                 # "put everything in the left bin": a default, explicit rules win
                    ev.all_bins.append(ps_cl[0])
                    ev.notes.append("all")
                carry_c, carry_p = [], []
                continue
            plain_p = [p for p in ps_cl if not isinstance(p, tuple)]
            if not cs_cl and not carry_c and plain_p and not is_neg and last_p is not None and set(plain_p) == {last_p} \
                    and "ALL" not in ks and "REST" not in ks and not ("WH" in ks and pending_keys):
                continue                              # apposition: "... the center bin, the one in the middle"
            if not cs_cl and not carry_c and ps_cl and pending_keys and not carry_p:
                # "... should not go to the center; put them on the left" / "not the left one but the right one"
                if is_neg:
                    ev.add([(k_, p) for k_ in pending_keys for p in ps_cl], neg=True)
                else:
                    ev.add([(k_, (plain_p or ps_cl)[-1]) for k_ in pending_keys])
                    pending_keys = []
                continue
            if pending_keys and not carry_c and not carry_p and not is_neg and cs_cl and items and items[0][0] == "P" \
                    and not isinstance(items[0][1], tuple) and (len(plain_p) >= 2 or sep == "contrast") \
                    and not any((x[0], x[1]) == (k_, items[0][1]) for x in ev.neg for k_ in pending_keys):
                ev.add([(k_, items[0][1]) for k_ in pending_keys])
                pending_keys = []
                items = items[1:]
            if carry_p and items and items[0][0] == "C" and not ps_cl:
                items = [("P", v) for v in dict.fromkeys(carry_p)] + items
            items = [("C", v) for v in carry_c] + items
            if neg_p and cs_cl and not is_neg:
                ev.add([(c_, p) for c_ in cs_cl for p in neg_p], neg=True)
            prev_carry_p = carry_p
            carry_c, carry_p = [], []
            if not items:
                continue
            pairs, lc, lp = _pair_runs(items, cross=is_neg)
            ev.add(pairs, neg=is_neg)
            if pairs:
                plain = [p for _, p in pairs if not isinstance(p, tuple)]
                last_p = plain[-1] if plain else last_p
                p_first = items[0][0] == "P" and not is_neg
                pending_keys = list(dict.fromkeys(c_ for c_, _ in pairs)) if is_neg else []
            elif lc and is_neg is False and not lp:
                pending_keys = []
            carry_c = list(dict.fromkeys(lc))
            carry_p = lp if not pairs else []
            carry_and = bool(lc) and not pairs and sep in ("and", "or", "ja", "und", "et", "y", "och", "e", "en") and not (set(ks) - {"C"})
            if pairs and not is_neg:
                neg_p = []
            if is_neg and lp and not pairs and not lc and ev.last_group and last_p is not None:   # "... and not the right one"
                ev.add([(g[0], p) for g in ev.last_group for p in lp if not isinstance(p, tuple)], neg=True)
                carry_p = []
            elif is_neg and lp and not pairs and not lc:
                neg_p += [p for p in lp if not isinstance(p, tuple)]
                carry_p = prev_carry_p
        if post_fix:
            ev.epoch += 1
            ev.add(post_fix)
            ev.notes.append("correction")
        if carry_p and not carry_c and term == "?" and not isinstance(carry_p[-1], tuple):
            q_p = carry_p[-1]                          # "Left bin? That's for blue."
        if carry_c and last_p is not None and ((p_first and term != "?") or any(k == "X" for k, _ in sent)):
            ev.add([(c_, last_p) for c_ in carry_c])
            ev.notes.append("carried")
        elif carry_c and not ps:
            pending_keys = list(carry_c) if len(set(carry_c)) < 3 else []
    return ev


def _solve(pos, neg, rel=(), keys=COLORS, fixed_prior=True):
    """Best of the 27 key->bin mappings against the evidence, a bijection prior and a very weak default-rule prior."""
    best, best_s, second = None, -1e9, -1e9
    for combo in itertools.product(BINS, repeat=3):
        m = dict(zip(keys, combo))
        s = 0.0
        for c_, p in pos:
            s += 3.0 if m.get(c_) == p else 0.0
        for c_, p in neg:
            s -= 3.0 if m.get(c_) == p else 0.0
        for c_, (_, kind, other) in rel:
            if c_ in m and other in m:
                d = BINS.index(m[c_]) - BINS.index(m[other])
                ok = {"l": d < 0, "r": d > 0, "n": abs(d) == 1, "s": d == 0}[kind]
                s += (3.0 if ok else 0.0) + (0.5 if ok and kind in "lr" and abs(d) == 1 else 0.0)
        if len(set(combo)) == 3:
            s += 1.0
        if fixed_prior:
            s += 0.05 * sum(m[c_] == FIXED[c_] for c_ in keys)
        if s > best_s:
            best, best_s, second = m, s, best_s
        elif s > second:
            second = s
    return best, best_s - second


def _resolve(ev, keys, fixed_prior):
    """Per key: (bin | None, tag). keys = COLORS or SHAPES."""
    pos = [(x[0], x[1]) for k in keys for x in ev.current(ev.pos, k)]
    neg = [(x[0], x[1]) for k in keys for x in ev.current(ev.neg, k)]
    rel = [(x[0], x[1]) for x in ev.rel if x[0] in keys and x[1][2] in keys]
    solved, _ = _solve(pos, neg, rel, keys, fixed_prior)
    out, tags = {}, {}
    for k in keys:
        leave = ev.current(ev.leave, k)
        kpos = ev.current(ev.pos, k)
        kneg = {p for _, p in neg if _ == k}
        if leave and (not kpos or max(x[3] for x in leave) > max(x[3] for x in kpos)):
            out[k], tags[k] = None, "leave"
            continue
        bins = list(dict.fromkeys(x[1] for x in kpos))
        good = [b for b in bins if b not in kneg]
        if len(good) == 1:
            out[k], tags[k] = good[0], "explicit" if len(bins) == 1 else "conflict"
        elif len(bins) >= 1:
            out[k], tags[k] = solved[k], "conflict"
    default = (ev.rest_bins[-1], "rest") if ev.rest_bins else ((ev.all_bins[-1], "all") if ev.all_bins else None)
    for k in keys:
        if k in out:
            continue
        kneg = {p for c_, p in neg if c_ == k}
        krel = [r for r in rel if r[0] == k]
        if krel and all(out.get(r[1][2]) is not None for r in krel):
            out[k], tags[k] = solved[k], "rel"
        elif default and default[0] not in kneg:
            out[k], tags[k] = default
        elif ev.rest_leave:
            out[k], tags[k] = None, "leave"
        elif len(kneg) == 2:
            out[k], tags[k] = [b for b in BINS if b not in kneg][0], "negelim"
    # the default rule, when the instruction refers to it: untouched colours keep their usual bin
    missing = [k for k in keys if k not in out]
    if missing and ev.usual and fixed_prior:
        taken = {v for v in out.values() if v}
        for k in missing:
            if FIXED[k] not in taken and FIXED[k] not in {p for c_, p in neg if c_ == k}:
                out[k], tags[k] = FIXED[k], "usual"
    # elimination: the two other keys sit in two different bins -> this one takes the third
    missing = [k for k in keys if k not in out]
    if len(missing) == 1:
        k = missing[0]
        others = [out[o] for o in keys if o != k]
        free = [b for b in BINS if b not in others and b not in {p for c_, p in neg if c_ == k}]
        named = k in ev.rembin or ev.rest_rembin
        if None not in others and len(set(others)) == 2 and len(free) == 1 and (named or (INFER_MISSING and not ev.only)):
            out[k], tags[k] = free[0], "rembin" if named else ("inferred" if k in ev.mentioned else "unmentioned")
    for k in keys:
        if k not in out:
            if fixed_prior and pos and not ev.only and not ev.rest_leave:
                out[k], tags[k] = solved[k], "guess"   # Parse.mapping keeps a guess; the rules abstain
            else:
                out[k], tags[k] = None, "unknown"
    return out, tags, pos, neg


def parse(text: str) -> Parse:
    toks = _tokens(text or "")
    force = [v for k, v in toks if k == "MODE"]
    real_h = {v for k, v in toks if k == "H"}
    block_is_cube = bool(real_h - {"cube"}) and "cube" not in real_h
    cleaned, skip_h = [], False
    for k, v in toks:
        if k == "ASIDE":                               # "whether cube or cylinder, ..." names no rule
            skip_h = True
            continue
        if k in ("C", "P", "S"):
            skip_h = False
        if k == "MODE" or (k == "H" and skip_h):
            continue
        if k == "HB":
            if not block_is_cube or skip_h:
                continue
            k = "H"
        cleaned.append((k, v))
    toks = cleaned
    if "coloronly" in force and "shapeonly" not in force:
        toks = [t for t in toks if t[0] != "H"]
    elif "shapeonly" in force and "coloronly" not in force and any(k == "H" for k, _ in toks) \
            and not any(k == "C" and isinstance(v, tuple) for k, v in _attach_shapes(toks)):
        toks = [t for t in toks if t[0] != "C"]
    has_c = any(k == "C" for k, _ in toks)
    has_h = any(k == "H" for k, _ in toks)
    notes = []
    mode = "color"
    if has_h and not has_c:
        mode = "shape"
        toks = [("C", v) if k == "H" else (k, v) for k, v in toks]
    elif has_h:
        merged = _attach_shapes(toks)
        bare = [v for k, v in merged if k == "C" and v in SHAPES]
        combos = [v for k, v in merged if k == "C" and isinstance(v, tuple)]
        catch_all = any(k in ("REST", "EXC") for k, _ in merged) or any(k == "LEAVE" for k, _ in merged)
        if _shape_subjects(merged, bare) or (combos and catch_all and _combos_matter(merged)) \
                or (combos and _combo_exception(merged)):
            mode, toks = "mixed", merged
        else:                                          # shapes are asides: sort by colour
            toks = [(k, v[0]) if k == "C" and isinstance(v, tuple) else (k, v) for k, v in merged
                    if not (k == "C" and v in SHAPES)]
    if mode == "mixed":
        return _parse_mixed(toks, notes)

    keys = SHAPES if mode == "shape" else COLORS
    ev = _collect(toks, keys)
    notes += ev.notes
    have = bool(ev.pos or ev.neg or ev.leave or ev.rest_bins or ev.all_bins or ev.rest_leave or ev.rel)
    if mode == "shape" and not have:                   # shapes are only listed ("every cube, cylinder and prism as usual")
        mode, keys = "color", COLORS
        toks = [t for t in toks if not (t[0] == "C" and t[1] in SHAPES)]
        ev = _collect(toks, keys)
        notes = list(ev.notes)
        have = bool(ev.pos or ev.neg or ev.leave or ev.rest_bins or ev.all_bins or ev.rest_leave or ev.rel)
    if have:
        out, tags, pos, neg = _resolve(ev, keys, mode == "color")
    else:
        pos, neg = [], []
        firsts = [a_ for a_, _ in ev.cc]
        if mode == "color" and len(ev.cc) >= 2 and len(set(firsts)) == len(firsts) and not ev.swaps and not ev.shift:
            out, tags = dict(FIXED), {k: "usual" for k in keys}
            for a_, b_ in ev.cc:
                out[a_], tags[a_] = FIXED[b_], ("rel" if ev.usual else "conflict")
            notes.append("colour-to-colour")
        elif mode == "color" and (ev.usual or ev.swaps or ev.reverse or ev.shift):
            out, tags = dict(FIXED), {k: "usual" for k in keys}
        elif mode == "color" and not any(k in ("C", "P", "H") for k, _ in toks):
            out, tags = dict(FIXED), {k: "norule" for k in keys}      # no rule words at all: the default rule
            notes.append("no-rule-words")
        else:
            out, tags = {k: None for k in keys}, {k: "unknown" for k in keys}
    if mode == "color":
        base_only = not [x for x in ev.pos if x[0] not in ev.kept]
        mirror_swap = any((kind == "C" and {a, b} == {"red", "blue"}) or (kind == "P" and {a, b} == {"left", "right"})
                          for kind, a, b in ev.swaps)
        if ev.reverse and base_only and ev.swaps and not mirror_swap:      # "mirror it, then yellow and blue swap"
            flip = {"left": "right", "right": "left", "center": "center", None: None}
            out = {c_: (p if c_ in ev.kept else flip[p]) for c_, p in (out if have else FIXED).items()}
        for kind, a, b in ev.swaps:
            if kind == "C" and a in out and b in out:
                if base_only:
                    out = dict(out) if have else dict(FIXED, **{k: v for k, v in out.items() if v})
                out[a], out[b] = out[b], out[a]
                tags[a] = tags[b] = "usual" if base_only else tags.get(a, "usual")
            elif kind == "P" and not isinstance(a, tuple) and not isinstance(b, tuple):
                inv = {a: b, b: a}
                out = {c_: inv.get(p, p) for c_, p in out.items()}
        if ev.shift:                                   # "every colour one bin to the left, wrapping around"
            for k in keys:
                if k not in ev.kept:
                    out[k], tags[k] = BINS[(BINS.index(FIXED[k]) + ev.shift) % 3], "rel"
        if ev.reverse and base_only and not ev.swaps:
            flip = {"left": "right", "right": "left", "center": "center", None: None}
            out = {c_: (p if c_ in ev.kept else flip[p]) for c_, p in out.items()}
        elif ev.reverse and not ev.swaps and have:     # "the usual way, only mirrored, yellow stays in the center"
            flip = {"left": "right", "right": "left", "center": "center"}
            firm = {v for k_, v in out.items() if v and tags.get(k_) != "usual"}
            for k in keys:
                if tags.get(k) == "usual" and k not in ev.kept and out.get(k) and flip[out[k]] not in firm:
                    out[k] = flip[out[k]]
    cert = {k: (tags[k], CERTAINTY[tags[k]]) for k in keys}
    raw = dict(out)
    final = {k: (v if cert[k][1] >= ABSTAIN_BELOW or v is None else None) for k, v in out.items()}
    conf = min(c for _, c in cert.values())
    pos2, neg2 = [(a, b) for a, b in pos], [(a, b) for a, b in neg]
    if mode == "shape":
        rules = dict(final)
        mapping = {c: (list(final.values())[0] if len(set(final.values())) == 1 else None) for c in COLORS}
        return Parse(mapping, conf, pos2, neg2, notes + ["by-shape"], rules, cert, True, raw)
    return Parse(raw, conf, pos2, neg2, notes, dict(final), cert, False, raw)


def _shape_subjects(toks, bare):
    """True if a shape without a colour is the subject of a bin: 'cubes go left', 'left bin: cylinders'."""
    if not bare:
        return False
    seq = [(k, v) for k, v in toks if k in ("C", "P", "S", "LEAVE")]
    for i, (k, v) in enumerate(seq):
        if k != "C" or v not in SHAPES:
            continue
        j = i + 1
        while j < len(seq) and seq[j][0] == "C" and seq[j][1] in SHAPES:
            j += 1
        if j < len(seq) and seq[j][0] in ("P", "LEAVE"):
            return True
        j = i - 1
        while j >= 0 and seq[j][0] == "C" and seq[j][1] in SHAPES:
            j -= 1
        if j >= 0 and seq[j][0] == "P" and (j == 0 or seq[j - 1][0] != "C"):
            return True
    return False


def _combo_exception(toks):
    """True if a (colour, shape) subject gets another bin than its colour, or two shapes of one colour differ:
    'blue goes left ... blue cubes go right instead' is a real shape rule, not an aside."""
    ev = _collect(toks, list(COLORS) + list(SHAPES))
    for c in COLORS:
        bare = {x[1] for x in ev.pos if x[0] == c}
        combo = {x[1] for x in ev.pos if isinstance(x[0], tuple) and x[0][0] == c}
        left = [x for x in ev.leave if isinstance(x[0], tuple) and x[0][0] == c]
        if (bare and combo - bare) or len(combo) > 1 or (left and (bare or combo)):
            return True
    return False


def _combos_matter(toks):
    """(colour, shape) subjects next to a catch-all are real rules unless every colour appears with one shape set
    only as an aside. Cheap test: some colour occurs both with a shape and in a catch-all sentence."""
    return any(k == "C" and isinstance(v, tuple) for k, v in toks)


def _parse_mixed(toks, notes):
    """Rules over colours, shapes and (colour, shape) pairs + a default, as the organisers' scorer resolves them."""
    universe = list(COLORS) + list(SHAPES)
    ev = _collect(toks, universe)
    keys = list(dict.fromkeys([x[0] for lst in (ev.pos, ev.neg, ev.leave) for x in lst]))
    rules, cert = {}, {}
    for k in keys:
        leave = ev.current(ev.leave, k)
        kpos = ev.current(ev.pos, k)
        kneg = {x[1] for x in ev.current(ev.neg, k)}
        if leave and (not kpos or max(x[3] for x in leave) > max(x[3] for x in kpos)):
            rules[k], cert[k] = None, ("leave", CERTAINTY["leave"])
            continue
        bins = list(dict.fromkeys(x[1] for x in kpos))
        good = [b for b in bins if b not in kneg]
        if len(good) == 1:
            tag = "explicit" if len(bins) == 1 else "conflict"
            rules[k], cert[k] = good[0], (tag, CERTAINTY[tag])
        elif bins:
            rules[k], cert[k] = bins[-1], ("conflict", CERTAINTY["conflict"])
        elif len(kneg) == 2:
            rules[k], cert[k] = [b for b in BINS if b not in kneg][0], ("negelim", CERTAINTY["negelim"])
        elif kneg and not isinstance(k, tuple):
            rules[k], cert[k] = None, ("unknown", 0.0)      # excluded from a general rule and nothing else said
        elif kneg:
            rules[k], cert[k] = None, ("leave", CERTAINTY["leave"])
    if ev.rest_bins or ev.all_bins:
        tag = "rest" if ev.rest_bins else "all"
        rules["default"], cert["default"] = (ev.rest_bins or ev.all_bins)[-1], (tag, CERTAINTY[tag])
    elif ev.rest_leave:
        rules["default"], cert["default"] = None, ("leave", CERTAINTY["leave"])
    # colour rules and shape rules that collide: the later statement is the exception
    order = {}
    for x in ev.pos + ev.leave:
        order[x[0]] = max(order.get(x[0], 0), x[3])
    for c in COLORS:
        for s in SHAPES:
            if c in rules and s in rules and (c, s) not in rules and rules[c] != rules[s]:
                rules[(c, s)] = rules[s] if order.get(s, 0) > order.get(c, 0) else rules[c]
                cert[(c, s)] = ("conflict", CERTAINTY["conflict"])
    final = {k: (v if cert[k][1] >= ABSTAIN_BELOW or v is None else None) for k, v in rules.items()}
    p = Parse({}, min([c for _, c in cert.values()] or [0.0]), [(x[0], x[1]) for x in ev.pos],
              [(x[0], x[1]) for x in ev.neg], notes + ev.notes + ["mixed"], final, cert, True, {})
    p.mapping = {c: (lambda t: t.pop() if len(t) == 1 else None)({p.target(c, s) for s in SHAPES}) for c in COLORS}
    p.raw = dict(p.mapping)
    return p


parse("Sort the objects: red into right, blue into left, yellow into center.")     # compile every regex at import

if __name__ == "__main__":
    import sys
    for line in sys.argv[1:] or sys.stdin:
        print(parse(line.strip()))
