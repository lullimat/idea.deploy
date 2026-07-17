# pymetallic patches for idea.deploy

Upstream: https://github.com/seantrue/pymetallic.git  
Pinned commit: `824e4714e791268faaae14565e778bd481b5a722` (v0.3.1)

## Patches (apply in order)

- `0001-recommended-max-working-set-size.patch` — expose
  `MTLDevice.recommendedMaxWorkingSetSize` via Swift FFI and
  `Device.recommended_max_working_set_size` in Python (used by idpy Metal
  `DiscoverGPUs` for `Memory`).

- `0002-library-compile-options.patch` — `Library(device, source, fast_math=True)`
  via `MTLCompileOptions.fastMathEnabled` and
  `metal_device_make_library_with_source_options` (idpy `optimizer_flag` parity).

## Regenerate

```bash
git clone https://github.com/seantrue/pymetallic.git /tmp/pymetallic-regen
git -C /tmp/pymetallic-regen checkout 824e4714e791268faaae14565e778bd481b5a722
# apply 0001, then desired 0002 edits, then:
git -C /tmp/pymetallic-regen diff > patches/pymetallic/0002-library-compile-options.patch
```

Applied automatically by `scripts/install-pymetallic.sh` (called from `idpy-init.sh`).
