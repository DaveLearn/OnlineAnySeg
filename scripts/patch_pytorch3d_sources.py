"""Guard pulsar's make_float3 fallback to CPU-only builds.

Under WITH_CUDA, pulsar's global.h includes ATen/cuda/CUDAContext.h, which pulls
in CUDA's vector_functions.h and its __host__ __device__
make_float3(float, float, float). pytorch3d then unconditionally defines its own
make_float3(const float&, const float&, const float&); the host compiler in this
toolchain finds the two equally good and every pulsar host call site fails with
"call of overloaded 'make_float3' is ambiguous". CUDA's version serves both host
and device, so the fallback is only needed for CPU-only builds. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD = """namespace py = pybind11;
inline float3 make_float3(const float& x, const float& y, const float& z) {
  float3 res;
  res.x = x;
  res.y = y;
  res.z = z;
  return res;
}
"""

NEW = """namespace py = pybind11;
#ifndef WITH_CUDA
// With CUDA, vector_functions.h already provides a __host__ __device__
// make_float3; defining this twin makes host calls ambiguous.
inline float3 make_float3(const float& x, const float& y, const float& z) {
  float3 res;
  res.x = x;
  res.y = y;
  res.z = z;
  return res;
}
#endif
"""


def main() -> None:
    p3d_root = Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/pytorch3d").resolve()
    path = p3d_root / "pytorch3d" / "csrc" / "pulsar" / "global.h"
    if not path.exists():
        raise SystemExit(f"Expected pytorch3d source file missing: {path}")
    text = path.read_text()
    if "#ifndef WITH_CUDA\n// With CUDA" in text:
        print("already patched: pulsar/global.h (make_float3 guard)", file=sys.stderr)
        return
    if OLD not in text:
        raise SystemExit("pulsar/global.h: expected make_float3 block not found; pytorch3d sources changed?")
    path.write_text(text.replace(OLD, NEW, 1))
    print("patched: pulsar/global.h (make_float3 guard)", file=sys.stderr)


if __name__ == "__main__":
    main()
