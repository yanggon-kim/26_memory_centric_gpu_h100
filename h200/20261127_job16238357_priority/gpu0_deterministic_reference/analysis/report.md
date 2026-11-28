# B8 deterministic HF diagnostic

The original B8worker returned after an exact reference failure. This is one separate same-input algorithm-control diagnostic, not an accepted tie or tolerance, a changed prompt/precision, a repaired default reference or a latency result. Preserve all dynamic and unchanged controlled StaticCache outputs.

- Limitation B8_deterministic_reference: One deterministic algorithm/workspace control failed or remained incomplete on the original exact B8inputs. Evidence: results/deterministic_HF_reference.json; results/processes/B8_deterministic_HF.json. Tried: Unchanged HF generate and controlled StaticCache/SDPA functions; deterministic Torch algorithms and explicit cuBLAS workspace,300second bound; all outputs retained. Resume: Retain B8as blocked and advance eligible full-GPU traffic/serving; investigate preserved scores separately without selecting favorable outputs/prompts or relaxing the default oracle.

Disposition: blocked_deterministic_reference. Original default reference remains failed; no quantitative comparison is promoted.

The deterministic control makes its six dynamic outputs repeat, but exact source/StaticCache agreement still fails. The first source mismatch moves to request5/token_index1, with a0.03125separate score margin (83vs67); an exact tie is not a sufficient explanation of all differences. No further reference retry is scheduled by default.
