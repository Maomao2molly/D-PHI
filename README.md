Implementation of D-PHI for minimization problems proposed in:

MaoMao Liang, Babooshka Shavazipour, Bhupinder Saini, and Michael Emmerich. 2026. D-PHI: Desirability-Based Hypervolume Indicator for Interactive Multiobjective Optimization Using Aspiration and Reservation Levels as
Preferences. ACM Trans. Evol. Learn. Optim. (2026). https://dl.acm.org/doi/10.1145/3794854


## Requirements

The implementation requires the following Python packages:

- numpy
- matplotlib
- moocore
- notebook (for running .ipynb example)
- ipykernel (for Jupyter kernel support)

---

## Repository Structure

```text
DPHI.py              # Core implementation of the D-PHI and CI indicator
contour.py           # Code to generate contour plots
toy_example.ipynb    # Example demonstrating how to compute D-PHI and CI
README.md            # This file
```

### Main class: D_PHI
Key methods:
- df(): Computes desirability function values for input objective vectors
- get_values(): Returns: D-PHI indicator value, CI (complementary indicator) value
- ASF_value(): Computes achievement scalarization function (ASF) values of each objective vectors for CI calculation

### Notes
- This implementation assumes a minimization problem
- Aspiration levels should be smaller than reservation levels