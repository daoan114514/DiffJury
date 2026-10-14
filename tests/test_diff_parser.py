import unittest

from diffjury.diff_parser import parse_hunks, parse_unified_diff


DIFF = """diff --git a/app.py b/app.py
index 123..456 100644
--- a/app.py
+++ b/app.py
@@ -2,3 +2,4 @@ def run():
 keep = True
-old = 1
+new = 2
+eval(user_input)
 tail = 3
"""


class DiffParserTests(unittest.TestCase):
    def test_parses_added_line_numbers(self):
        parsed = parse_unified_diff(DIFF)
        self.assertEqual(["app.py"], parsed.files)
        self.assertEqual([(3, "new = 2"), (4, "eval(user_input)")], [(x.line, x.content) for x in parsed.added_lines])

    def test_removed_double_dash_and_added_double_plus_lines_keep_numbers(self):
        diff = (
            "--- a/q.sql\n+++ b/q.sql\n@@ -1,3 +1,3 @@\n"
            " select 1;\n"
            "--- old comment\n"
            "+++i;\n"
            " select 2;\n"
        )
        parsed = parse_unified_diff(diff)
        self.assertEqual(["q.sql"], parsed.files)
        self.assertEqual([("q.sql", 2, "++i;")], [
            (x.path, x.line, x.content) for x in parsed.added_lines
        ])
        hunk = parse_hunks(diff)[0]
        self.assertEqual(["select 1;", "-- old comment", "++i;", "select 2;"], [
            line.text for line in hunk.lines
        ])
        self.assertEqual(3, hunk.lines[-1].new_line)

    def test_new_deleted_and_multiple_files(self):
        diff = (
            "diff --git a/new.py b/new.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n"
            "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n"
            "--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
            "diff --git a/lib/c.py b/lib/c.py\n--- a/lib/c.py\n+++ b/lib/c.py\n"
            "@@ -10 +10 @@\n-y = 0\n+y = 1\n\\ No newline at end of file\n"
        )
        parsed = parse_unified_diff(diff)
        self.assertEqual(["new.py", "lib/c.py"], parsed.files)
        self.assertEqual([("new.py", 1), ("new.py", 2), ("lib/c.py", 10)], [
            (x.path, x.line) for x in parsed.added_lines
        ])
        self.assertEqual(3, len(parse_hunks(diff)))

    def test_numbered_rendering_uses_new_file_line(self):
        lines = parse_hunks(DIFF)[0].lines
        self.assertEqual("+   4 | eval(user_input)", lines[3].numbered)
        self.assertEqual("-     | old = 1", lines[1].numbered)


if __name__ == "__main__":
    unittest.main()
