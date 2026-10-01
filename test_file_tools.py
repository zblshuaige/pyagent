import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class FileToolsTest(unittest.TestCase):
	def setUp(self):
		self.temporary_directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.temporary_directory.cleanup)
		output_patch = patch.object(app, "_OUTPUT_DIR", Path(self.temporary_directory.name))
		output_patch.start()
		self.addCleanup(output_patch.stop)

	def test_create_read_and_rewrite(self):
		filename = "report.md"
		self.assertEqual(
			app.create_text_file.invoke({"filename": filename, "content": "初稿"}),
			"Created outputs/report.md",
		)
		self.assertEqual(app.read_text_file.invoke({"filename": filename}), "初稿")
		self.assertIn("Error:", app.create_text_file.invoke({"filename": filename, "content": "覆盖"}))
		self.assertIn(
			"Error:",
			app.rewrite_text_file.invoke(
				{"filename": filename, "original_content": "旧数据", "updated_content": "破坏性改写"}
			),
		)
		self.assertEqual(app.read_text_file.invoke({"filename": filename}), "初稿")
		self.assertEqual(
			app.rewrite_text_file.invoke(
				{"filename": filename, "original_content": "初稿", "updated_content": "定稿"}
			),
			"Updated outputs/report.md",
		)
		self.assertEqual(app.read_text_file.invoke({"filename": filename}), "定稿")

	def test_replace_only_one_matching_snippet(self):
		filename = "notes.txt"
		app.create_text_file.invoke({"filename": filename, "content": "待办\n待办\n结束"})
		self.assertIn(
			"Error:",
			app.replace_text_file.invoke({"filename": filename, "old_text": "待办", "new_text": "完成"}),
		)
		self.assertIn(
			"Error:",
			app.replace_text_file.invoke({"filename": filename, "old_text": "", "new_text": "完成"}),
		)
		self.assertEqual(app.read_text_file.invoke({"filename": filename}), "待办\n待办\n结束")
		self.assertEqual(
			app.replace_text_file.invoke({"filename": filename, "old_text": "结束", "new_text": "完成"}),
			"Updated outputs/notes.txt",
		)
		self.assertEqual(app.read_text_file.invoke({"filename": filename}), "待办\n待办\n完成")

	def test_accept_text_files_with_any_extension(self):
		for filename in ("birthday_cake.html", "style.css", "data.json", "script.py", "notes"):
			with self.subTest(filename=filename):
				self.assertEqual(
					app.create_text_file.invoke({"filename": filename, "content": "草稿"}),
					f"Created outputs/{filename}",
				)
				self.assertEqual(app.read_text_file.invoke({"filename": filename}), "草稿")
				self.assertEqual(
					app.replace_text_file.invoke({"filename": filename, "old_text": "草稿", "new_text": "定稿"}),
					f"Updated outputs/{filename}",
				)
				self.assertEqual(app.read_text_file.invoke({"filename": filename}), "定稿")

	def test_reject_out_of_scope_paths(self):
		for filename in ("../app.py", "folder/report.md", "C:\\temp\\report.md", ".", ".."):
			with self.subTest(filename=filename):
				self.assertIn("Error:", app.read_text_file.invoke({"filename": filename}))
				self.assertIn(
					"Error:",
					app.create_text_file.invoke({"filename": filename, "content": "bad"}),
				)
				self.assertIn(
					"Error:",
					app.replace_text_file.invoke(
						{"filename": filename, "old_text": "old", "new_text": "new"}
					),
				)

	def test_reject_excessive_content(self):
		self.assertIn(
			"Error:",
			app.create_text_file.invoke(
				{"filename": "large.txt", "content": "x" * (app._MAX_FILE_CHARS + 1)}
			),
		)
		self.assertFalse((Path(self.temporary_directory.name) / "large.txt").exists())


if __name__ == "__main__":
	unittest.main()