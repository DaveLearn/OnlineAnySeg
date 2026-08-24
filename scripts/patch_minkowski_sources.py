"""Add thrust includes MinkowskiEngine relies on transitively.

The thrust shipped with CUDA >= 11.6 no longer includes execution policies /
algorithms transitively, so unmodified ME v0.5.4 fails with errors like
``namespace "thrust" has no member "device"`` (upstream issues #543 / #621,
unfixed -- ME is unmaintained). Inserting the explicit includes is the
canonical fix. Idempotent: run before every build.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCHES: dict[str, list[str]] = {
    "src/3rdparty/concurrent_unordered_map.cuh": [
        "thrust/execution_policy.h",
    ],
    "src/convolution_kernel.cuh": [
        "thrust/execution_policy.h",
    ],
    "src/coordinate_map_gpu.cu": [
        "thrust/execution_policy.h",
        "thrust/unique.h",
        "thrust/remove.h",
        "thrust/sort.h",
    ],
    "src/spmm.cu": [
        "thrust/execution_policy.h",
        "thrust/reduce.h",
        "thrust/sort.h",
    ],
}


def patch_file(path: Path, headers: list[str]) -> bool:
    text = path.read_text()
    missing = [h for h in headers if f"#include <{h}>" not in text]
    if not missing:
        return False

    lines = text.splitlines(keepends=True)
    insert_at = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#include")), 0)
    lines[insert_at:insert_at] = [f"#include <{h}>\n" for h in missing]
    path.write_text("".join(lines))
    return True


UNIQUE_TO_SHARED_OLD = """      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
"""

UNIQUE_TO_SHARED_NEW = """      // gcc's shared_ptr(unique_ptr&&) converter calls unqualified __to_address,
      // which is ambiguous under ADL here: thrust::pair is cuda::std::pair on
      // CUDA >= 11.6, so cuda::std is an associated namespace of map_type* and
      // libcu++ declares its own __to_address. The raw-pointer + deleter
      // constructor takes the same ownership without that call.
      {
        auto unique_map = map_type::create(
            compute_hash_table_size(size, m_hashtable_occupancy),
            m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
        auto map_deleter = unique_map.get_deleter();
        m_map = std::shared_ptr<map_type>(unique_map.release(), std::move(map_deleter));
      }
"""


def patch_unique_to_shared(me_root: Path) -> None:
    path = me_root / "src/coordinate_map_gpu.cuh"
    text = path.read_text()
    if "unique_map.release()" in text:
        print("already patched: src/coordinate_map_gpu.cuh (shared_ptr conversion)", file=sys.stderr)
        return
    if UNIQUE_TO_SHARED_OLD not in text:
        raise SystemExit("coordinate_map_gpu.cuh: expected m_map assignment not found; ME sources changed?")
    path.write_text(text.replace(UNIQUE_TO_SHARED_OLD, UNIQUE_TO_SHARED_NEW, 1))
    print("patched: src/coordinate_map_gpu.cuh (shared_ptr conversion)", file=sys.stderr)


def main() -> None:
    me_root = Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/MinkowskiEngine").resolve()
    if not me_root.is_dir():
        raise SystemExit(f"MinkowskiEngine root not found: {me_root}")

    for rel_path, headers in PATCHES.items():
        path = me_root / rel_path
        if not path.exists():
            raise SystemExit(f"Expected ME source file missing: {path}")
        changed = patch_file(path, headers)
        print(f"{'patched' if changed else 'already patched'}: {rel_path}", file=sys.stderr)

    patch_unique_to_shared(me_root)


if __name__ == "__main__":
    main()
