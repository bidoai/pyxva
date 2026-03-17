from __future__ import annotations

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.schedule import Schedule


class InterestRateSwap(Pricer):
    """Plain vanilla interest rate swap (fixed vs floating).

    The payer swap (long fixed, receive floating) MTM at time t is:
    V(t) = PV(floating leg) - PV(fixed leg)
         = N · [P(t, t_0) - P(t, T_N) - K · Σ δ_i · P(t, T_i)]

    Uses simplified flat-curve discount factors from the short rate r(t).

    Parameters
    ----------
    fixed_rate : float
        Fixed leg coupon rate.
    maturity : float | None
        Swap maturity in years. Required when ``schedule`` is None.
    notional : float
        Notional principal.
    payment_freq : int
        Payments per year (e.g. 4 = quarterly). Used only when ``schedule``
        is None.
    payer : bool
        True = payer (pay fixed, receive floating); False = receiver.
    schedule : Schedule | None
        Pre-built payment schedule with calendar- and day-count-adjusted
        payment times and accrual fractions. When provided, ``maturity``
        and ``payment_freq`` are ignored and the schedule's ``payment_times``
        and ``day_count_fractions`` are used instead.
    """

    def __init__(
        self,
        fixed_rate: float,
        maturity: float | None = None,
        notional: float = 1_000_000.0,
        payment_freq: int = 4,
        payer: bool = True,
        schedule: Schedule | None = None,
    ) -> None:
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.payer = payer
        self.schedule = schedule

        if schedule is not None:
            self.payment_times = schedule.payment_times          # (n,)
            self.deltas = schedule.day_count_fractions           # (n,) — δᵢ per period
            self.maturity = float(schedule.payment_times[-1])
        else:
            if maturity is None:
                raise ValueError("Either maturity or schedule must be provided.")
            self.maturity = maturity
            self.payment_freq = payment_freq
            dt = 1.0 / payment_freq
            n = int(round(maturity * payment_freq))
            self.payment_times = np.array([dt * (i + 1) for i in range(n)])
            self.deltas = np.full(n, dt)                         # uniform δ

    def price(self, result: SimulationResult) -> np.ndarray:
        """Compute swap MTM at each time step on each path.

        Uses the standard annuity formula:
        V_payer(t) = N · [(1 - P(t, T_N)) - K · A(t)]

        where A(t) = Σ_{T_i > t} δ · P(t, T_i)  (annuity factor)
        P(t, T) ≈ exp(-r(t) · (T - t))           (flat-curve approximation)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        r = result.factor("r")  # (n_paths, T)
        time_grid = result.time_grid
        n_paths, n_steps = r.shape
        mtm = np.zeros((n_paths, n_steps))

        future_mask = self.payment_times > 0  # updated per step below
        for i, t in enumerate(time_grid):
            future_mask = self.payment_times > t
            if not future_mask.any():
                continue

            r_t = r[:, i]                                     # (n_paths,)
            future_T = self.payment_times[future_mask]        # (k,)
            future_delta = self.deltas[future_mask]           # (k,)

            # Annuity: Σ δᵢ · P(t, Tᵢ)  — vectorised over payments
            tau = future_T - t                                # (k,)
            df = np.exp(-r_t[:, None] * tau[None, :])        # (n_paths, k)
            annuity = (future_delta[None, :] * df).sum(axis=1)  # (n_paths,)

            # Final discount P(t, T_N)
            tau_N = self.maturity - t
            P_tN = np.exp(-r_t * tau_N) if tau_N > 0 else np.ones(n_paths)

            swap_value = self.notional * ((1 - P_tN) - self.fixed_rate * annuity)
            mtm[:, i] = swap_value if self.payer else -swap_value

        return mtm
