import unittest
import tempfile
from pathlib import Path

import sys
sys.path.append("./")

from utils.files import _copy_folder


class TestCopyFolder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the source and destination
        self.temp_dir = tempfile.TemporaryDirectory()
        self.src_dir = Path(self.temp_dir.name, "io/src")
        self.dst_dir = Path(self.temp_dir.name, "io/dst")
        self.base_src_dir = self.src_dir
        
        # Setup initial directory structure for tests
        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Test1: File in src root
        (self.src_dir / "file1.py").touch()
        
        # Test2: Directory in src
        (self.src_dir / "abc").mkdir()
        
        # Test3: Subdirectory in src
        (self.src_dir / "abc/bcd").mkdir()
        
        # Test4: File in subdirectory
        (self.src_dir / "abc/file_abc.py").touch()

    def tearDown(self):
        # Cleanup the temporary directory after tests
        self.temp_dir.cleanup()

    def test_copy_file_root(self):
        src = str(self.src_dir / "file1.py")
        _copy_folder(src, str(self.dst_dir), str(self.base_src_dir))
        self.assertTrue((self.dst_dir / "file1.py").exists())

    def test_copy_directory(self):
        src = str(self.src_dir / "abc")
        _copy_folder(src, str(self.dst_dir), str(self.base_src_dir))
        self.assertTrue((self.dst_dir / "abc").exists())

    def test_copy_subdirectory(self):
        src = str(self.src_dir / "abc/bcd")
        _copy_folder(src, str(self.dst_dir), str(self.base_src_dir))
        self.assertTrue((self.dst_dir / "abc/bcd").exists())

    def test_copy_file_in_subdirectory(self):
        src = str(self.src_dir / "abc/file_abc.py")
        _copy_folder(src, str(self.dst_dir), str(self.base_src_dir))
        self.assertTrue((self.dst_dir / "abc/file_abc.py").exists())

if __name__ == "__main__":
    unittest.main()
