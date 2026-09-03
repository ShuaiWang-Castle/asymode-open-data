# Flow-selection phase-diagram precheck

This is a local, fixed-design Monte Carlo audit of the core one-flow-versus-two-flow theory. It is not a neural-network result and is not part of the real-data main table.

Configuration: `D=R=0.05`, `mu=0.5`, `sigma=0.05`, `5000` Monte Carlo replicates per cell.

## Integrity checks

- constrained oracle-gap identity: 100,000 random distributions, maximum absolute error `5.551e-17`;
- `det(Q)=v`: maximum absolute error `7.778e-16`;
- fixed-design risk formulas: maximum Monte Carlo absolute error `4.365e-06`;
- cells with `|Gamma_n-1|>0.15`: 25;
- theoretical/Monte Carlo side-of-threshold mismatches: 0.

## Selected cells

| n | v | Gamma_n | one-flow risk MC | two-flow risk MC | theory preference |
|---:|---:|---:|---:|---:|---|
| 16 | 0.0000 | 0.000 | 0.000157 | 0.000157 | tie / rank one |
| 16 | 0.0016 | 0.102 | 0.000172 | 0.000310 | one flow |
| 16 | 0.0064 | 0.399 | 0.000217 | 0.000310 | one flow |
| 16 | 0.0256 | 1.486 | 0.000390 | 0.000314 | two flow |
| 32 | 0.0000 | 0.000 | 0.000079 | 0.000079 | tie / rank one |
| 32 | 0.0016 | 0.203 | 0.000095 | 0.000157 | one flow |
| 32 | 0.0064 | 0.799 | 0.000141 | 0.000154 | one flow |
| 32 | 0.0256 | 2.972 | 0.000312 | 0.000157 | two flow |
| 64 | 0.0000 | 0.000 | 0.000040 | 0.000040 | tie / rank one |
| 64 | 0.0016 | 0.407 | 0.000054 | 0.000076 | one flow |
| 64 | 0.0064 | 1.598 | 0.000102 | 0.000078 | two flow |
| 64 | 0.0256 | 5.945 | 0.000270 | 0.000077 | two flow |
| 128 | 0.0000 | 0.000 | 0.000019 | 0.000019 | tie / rank one |
| 128 | 0.0016 | 0.814 | 0.000036 | 0.000040 | one flow |
| 128 | 0.0064 | 3.195 | 0.000082 | 0.000038 | two flow |
| 128 | 0.0256 | 11.890 | 0.000251 | 0.000038 | two flow |
| 256 | 0.0000 | 0.000 | 0.000010 | 0.000010 | tie / rank one |
| 256 | 0.0016 | 1.628 | 0.000026 | 0.000020 | two flow |
| 256 | 0.0064 | 6.390 | 0.000072 | 0.000020 | two flow |
| 256 | 0.0256 | 23.779 | 0.000242 | 0.000020 | two flow |

## Reading

The controlled crossover occurs at `Gamma_n = n G_n / sigma^2 = 1`, up to Monte Carlo error. When `v=0`, the two-column design has rank one, the two flows are not separately identifiable, and the two fitted-mean spaces coincide. Increasing either state dispersion or sample size moves the same physical two-flow system from the parsimonious one-flow regime into the two-flow regime.
