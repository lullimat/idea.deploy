# pymetallic patches for idea.deploy

Upstream: https://github.com/seantrue/pymetallic.git  
Pinned commit: `824e4714e791268faaae14565e778bd481b5a722` (v0.3.1)

## Patches

- `0001-recommended-max-working-set-size.patch` — expose
  `MTLDevice.recommendedMaxWorkingSetSize` via Swift FFI and
  `Device.recommended_max_working_set_size` in Python (used by idpy Metal
  `DiscoverGPUs` for `Memory`).

## Regenerate

```bash
git clone https://github.com/seantrue/pymetallic.git /tmp/pymetallic-regen
git -C /tmp/pymetallic-regen checkout 824e4714e791268faaae14565e778bd481b5a722
# apply desired edits, then:
git -C /tmp/pymetallic-regen diff > patches/pymetallic/0001-recommended-max-working-set-size.patch
```

Applied automatically by `scripts/install-pymetallic.sh` (called from `idpy-init.sh`).
