# Estimation filters and analysis

Standalone attitude filters (recurrence blocks, siblings of each other)
and the observability / consistency analysis tools.

## Attitude filters

::: manta.Madgwick

::: manta.Mahony

::: manta.IMUIntegrator

::: manta.IMUPreintegrator

## Inertial navigation

See the [standalone INS guide](../how-to/use-ins.md) for an example using your own model.

The low-level [INS transform][manta.INS] also supports custom packet scheduling.

::: manta.INSStream

::: manta.NavigationEnvironment

::: manta.NavigationFrame

::: manta.PreintegratedIMUPacket

## Observability and consistency

::: manta.estimation.observability

::: manta.estimation.observability_trajectory

::: manta.estimation.sigma_horizon

::: manta.estimation.nees

::: manta.estimation.chi2_gate

::: manta.estimation.chi2_quantile
