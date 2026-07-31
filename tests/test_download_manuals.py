import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.download_public_manuals as download_module
from scripts.download_public_manuals import (
    load_checksums,
    safe_filename,
    verify_pdf,
)


class DownloadManualTests(unittest.TestCase):
    def test_failed_download_does_not_replace_existing_manual(self) -> None:
        existing = b"%PDF-1.7\nknown-good\n"
        downloaded = b"%PDF-1.7\nunexpected\n"

        class Response(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "manual.pdf"
            destination.write_bytes(existing)
            expected = {
                "size_bytes": len(existing),
                "sha256": hashlib.sha256(existing).hexdigest(),
            }
            with patch.object(
                download_module,
                "urlopen",
                return_value=Response(downloaded),
            ):
                with self.assertRaises(ValueError):
                    download_module.download_one(
                        "https://example.com/manual.pdf",
                        destination,
                        10,
                        expected,
                        require_checksum=True,
                    )
            self.assertEqual(destination.read_bytes(), existing)

    def test_verified_pdf_matches_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "manual.pdf"
            content = b"%PDF-1.7\nminimal fixture\n"
            pdf.write_bytes(content)
            expected = {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            size, digest = verify_pdf(
                pdf,
                expected,
                require_checksum=True,
            )
            self.assertEqual(size, len(content))
            self.assertEqual(digest, expected["sha256"])

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / "manual.pdf"
            pdf.write_bytes(b"%PDF-1.7\nfixture\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_pdf(
                    pdf,
                    {"size_bytes": pdf.stat().st_size, "sha256": "0" * 64},
                    require_checksum=True,
                )

    def test_unsafe_or_non_pdf_filename_is_rejected(self) -> None:
        for value in ("../manual.pdf", "manual.txt", "", "a/b.pdf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    safe_filename(value)

    def test_checksum_lock_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checksums.json"
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_checksums(path)


if __name__ == "__main__":
    unittest.main()
