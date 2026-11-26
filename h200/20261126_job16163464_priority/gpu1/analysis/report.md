# GPU 1: GPU-b51c6186-ac3b-5149-8b8a-f33943072c47

- Limitation P0_native_counter_export: Initial read-only NCU CSV export used unsupported --units option; capture itself completed successfully. Evidence: profiles/P0_native_ncu_stream512_export_units_v1.failed.log; profiles/P0_native_ncu_stream512.ncu-repz. Tried: Preserved failed export; checked local ncu --help and reran import with supported --print-units base. Production traffic adapter already uses this supported flag.. Resume: Qualify corrected CSV metric rows before using counter values; no GPU recapture required.
