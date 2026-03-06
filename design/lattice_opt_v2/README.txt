
Lattice Optimization V2 (damage fixes)
======================================

What changed vs V1
------------------
1. Damage solve no longer blindly attempts to solve singular systems.
2. After damage, only the root-connected component is retained.
3. If no root-connected structure remains, the design is marked as failed.
4. Small stiffness regularization is added to improve numerical robustness.
5. Pruning thresholds were made less conservative so pruning can actually happen.

Run
---
pip install numpy scipy matplotlib networkx
python optimize_v2_fixed.py

What to look for
----------------
- No more NaN damaged results
- damage_reason in console output
- possible reduction in edge count after a few iterations
- meaningful damaged λ2 history
