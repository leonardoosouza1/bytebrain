"""Unit tests for addressing.py (pure CPU, no sandbox, no model). Run: python -m unittest test_addressing"""

import ast
import random
import unittest

import addressing as A
import fixes as F

SRC = F.load_generated()
TEXTS = ["return 0", "pass", "x = 1\nreturn x", "return (", "def f(:\n    pass", '"""open', "",
         "        return 1", "if True:\nreturn 2", "return 1\n  return 2", "raise ValueError('x')",
         "for i in range(3):\n    pass\nreturn i"]


def dump(t):
    return ast.dump(t, include_attributes=True)


class TestPatch(unittest.TestCase):
    def test_fast_equals_full(self):
        """apply_patch_fast must accept/reject exactly like apply_patch and yield the same source/tree."""
        rng = random.Random(0)
        lines = SRC.splitlines()
        n_cases = n_acc = 0
        for _ in range(300):
            ln = rng.randint(1, len(lines))
            loc = A.locate(SRC, ln)
            if loc is None:
                continue
            addr = rng.choice(list(loc["addresses"].values()))
            text = rng.choice(TEXTS)
            try:
                full = A.apply_patch(SRC, addr, text)
            except A.PatchRejected:
                full = None
            try:
                fast = A.apply_patch_fast(SRC, addr, text, ast.parse(SRC))
            except A.PatchRejected:
                fast = None
            n_cases += 1
            self.assertEqual(full is None, fast is None, (addr, text))
            if full:
                n_acc += 1
                self.assertEqual(full[0], fast[0])
                self.assertEqual(dump(ast.parse(full[0])), dump(fast[1]))
        self.assertGreater(n_cases, 200)
        self.assertGreater(n_acc, 30)

    def test_rejects(self):
        tree = ast.parse(SRC)
        gi = A.functions(tree)["gcd"][0]
        bad = [
            ({"kind": "symbol", "name": "gcd", "stmt": None}, "def gcd2(a, b):\n    return 0"),  # rename
            ({"kind": "symbol", "name": "gcd", "stmt": None}, "def gcd(a):\n    return 0"),  # re-sign
            ({"kind": "path", "path": f"body[{gi}]"}, ""),  # delete function
            ({"kind": "path", "path": f"body[{gi}].body[9]"}, "pass"),  # out of range
            ({"kind": "path", "path": "body[1].__class__"}, "pass"),  # malformed path
            ({"kind": "symbol", "name": "nope", "stmt": [0, 1]}, "pass"),  # unknown symbol
            ({"kind": "symbol", "name": "gcd", "stmt": [0, 2]}, ""),  # empty body
            ({"kind": "symbol", "name": "gcd", "stmt": [1, 2]}, "return (a"),  # syntax
            ({"kind": "span", "start": [9999, 0], "end": [9999, 1]}, "x"),  # span out of range
        ]
        for addr, text in bad:
            with self.assertRaises(A.PatchRejected, msg=(addr, text)):
                A.apply_patch(SRC, addr, text)

    def test_reindent_any_indent(self):
        tree = ast.parse(SRC)
        gi = A.functions(tree)["gcd"][0]
        addr = {"kind": "path", "path": f"body[{gi}].body[1].body[0]"}
        a, _ = A.apply_patch(SRC, addr, "return abs(a)")
        b, _ = A.apply_patch(SRC, addr, "            return abs(a)\n")
        self.assertEqual(a, b)
        self.assertIn("        return abs(a)\n", a)

    def test_insertion(self):
        new, _ = A.apply_patch(SRC, {"kind": "symbol", "name": "factorial", "stmt": [1, 1]}, "n = n")
        self.assertEqual(new.count("\n"), SRC.count("\n") + 1)

    def test_locate_roundtrip(self):
        loc = A.locate(SRC, 523)
        self.assertEqual(loc["symbol"], "gcd")
        s, e, _ = A.resolve(SRC, loc["addresses"]["path_deep"])
        self.assertEqual(SRC[s:e].strip(), "return a")
        s, e, _ = A.resolve(SRC, loc["addresses"]["span_line"])
        self.assertEqual(SRC[s:e], "return a")


if __name__ == "__main__":
    unittest.main()
