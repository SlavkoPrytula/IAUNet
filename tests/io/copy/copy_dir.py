from utils.files import _copy_folder


if __name__ == "__main__":
    # test1
    # src = "tests/io/src/file1.py"
    # dst = "tests/io/dst"

    # test2
    src = "tests/io/src/abc/"
    dst = "tests/io/dst"

    # test3
    # src = "tests/io/src/abc/bcd"
    # dst = "tests/io/dst"

    # test4
    # src = "tests/io/src/abc/file_abc.py"
    # dst = "tests/io/dst"

    base_src_dir = "tests/io/src"
    _copy_folder(src, dst, base_src_dir)
