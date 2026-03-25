"""SA-CCR (Standardised Approach for Counterparty Credit Risk) calculator.

Implements the full BCBS 279 / Basel III CRE52 SA-CCR formula, including
Annex B supervisory duration and three-bucket IR aggregation:

    EAD = alpha × (RC + AddOn_aggregate)

where:
    alpha = 1.4
    RC    = max(V_net - C, 0)   — replacement cost (net MTM less collateral)
    AddOn = sum of asset-class add-ons

Interest-rate add-on (Annex B / CRE52):
    1. Supervisory duration: SD_j = (exp(-0.05*S_j) - exp(-0.05*E_j)) / 0.05
    2. Adjusted notional:    d_j  = Notional_j × SD_j
    3. Maturity factor:      MF_j = sqrt(min(M_j, 1Y))  [unmargined]
                             MF_j = 1.5 × sqrt(MPOR)    [margined]
    4. Effective notional per bucket b, per currency:
           D_b^ccy = sum_j (delta_j × d_j × MF_j)   for j in bucket b
    5. Three-bucket aggregation per currency (rho12=rho23=0.70, rho13=0.30):
           AddOn_IR^ccy = sqrt(
               (D1^ccy)^2 + (D2^ccy)^2 + (D3^ccy)^2
               + rho12*(D1^ccy*D2^ccy + D2^ccy*D3^ccy)    [ρ₁₂=ρ₂₃ combined]
               + rho13*(D1^ccy*D3^ccy)
           )
    6. Cross-currency sum (no netting across currencies):
           AddOn_IR = sum_ccy AddOn_IR^ccy
    7. SF_IR = 0.005 applied to the aggregate:
           AddOn_IR_final = SF_IR × AddOn_IR

Non-IR asset classes use the supervisory-factor approach (CRE52.72).

Reference: Basel Committee on Banking Supervision, "The standardised approach
for measuring counterparty credit risk exposures", March 2014 (rev. April 2014),
BCBS 279 — see CRE52 and Annex B for IR hedging-set details.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Supervisory factor table (Basel III CRE52.72)
# ---------------------------------------------------------------------------

# IR supervisory factor — single value used in the full Annex B approach.
# (The old per-bucket IR factors below are retained for reference only.)
_IR_SF = 0.005   # CRE52.72 supervisory factor for IR hedging sets

# Legacy IR supervisory factors by maturity bucket (kept for reference).
_IR_FACTORS = {
    "lt1y":       0.0020,   # < 1 year
    "1y5y":       0.0050,   # 1–5 years
    "gt5y_lt10y": 0.0100,   # 5–10 years
    "gt10y":      0.0150,   # > 10 years
}

# Non-IR supervisory factors (CRE52.72)
_SF = {
    "equity_single":    0.32,
    "equity_index":     0.20,
    "fx":               0.04,
    "commodity_energy": 0.18,
    "commodity_metals": 0.18,
    "commodity_other":  0.18,
    "credit_ig":        0.0038,
    "credit_sg":        0.0106,
}

# Alpha multiplier (CRE52.47)
_ALPHA = 1.4

# Annex B three-bucket correlation parameters
_RHO_12 = 0.70   # between bucket 1 and bucket 2
_RHO_23 = 0.70   # between bucket 2 and bucket 3
_RHO_13 = 0.30   # between bucket 1 and bucket 3


# ---------------------------------------------------------------------------
# Stand-alone helper functions (Annex B)
# ---------------------------------------------------------------------------

def supervisory_duration(start: float, end: float) -> float:
    """BCBS 279 Annex B supervisory duration (SD).

    SD = (exp(-0.05 * S) - exp(-0.05 * E)) / 0.05

    Parameters
    ----------
    start : float
        Start date of the accrual/coupon period in years (S_j).
    end : float
        End date of the accrual/coupon period in years (E_j).

    Returns
    -------
    float
        Supervisory duration.  Returns 1.0 for a degenerate case where
        start == end (avoids division by zero).
    """
    if abs(end - start) < 1e-10:
        return 1.0  # degenerate: treat as unit duration
    # Annex B formula — note sign: exp(-0.05*S) > exp(-0.05*E) when S < E,
    # so SD is positive for a normal (S < E) instrument.
    return (math.exp(-0.05 * start) - math.exp(-0.05 * end)) / 0.05


def maturity_factor(
    maturity: float,
    is_margined: bool = False,
    mpor: float = 10 / 252,
) -> float:
    """BCBS 279 Annex B maturity factor (MF).

    Unmargined: MF = sqrt(min(M, 1Y) / 1Y)
    Margined:   MF = 1.5 × sqrt(MPOR / 1Y)

    Note: the /1Y denominator normalises units; since all inputs are already
    in years the formula simplifies to sqrt(min(M, 1)) and 1.5×sqrt(MPOR).

    Parameters
    ----------
    maturity : float
        Residual maturity of the trade in years.
    is_margined : bool
        True for a margined netting set (uses MPOR).
    mpor : float
        Margin period of risk in years (default 10 business days ≈ 10/252).

    Returns
    -------
    float
        Maturity factor (dimensionless).
    """
    if is_margined:
        # CRE52 margined formula: 1.5 × sqrt(MPOR)
        return 1.5 * math.sqrt(mpor)
    # CRE52 unmargined formula: sqrt(min(M, 1Y))
    return math.sqrt(min(maturity, 1.0))


def _ir_bucket(maturity: float) -> int:
    """Return the Annex B IR maturity bucket (1, 2, or 3) for a given residual maturity.

    Bucket 1: maturity < 1 year
    Bucket 2: 1 year <= maturity <= 5 years
    Bucket 3: maturity > 5 years
    """
    if maturity < 1.0:
        return 1
    elif maturity <= 5.0:
        return 2
    else:
        return 3


def _ir_supervisory_factor(maturity: float) -> float:
    """Return the legacy IR supervisory factor for a given maturity (years).

    Kept for internal backward-compatibility use only.
    """
    if maturity < 1.0:
        return _IR_FACTORS["lt1y"]
    elif maturity <= 5.0:
        return _IR_FACTORS["1y5y"]
    elif maturity <= 10.0:
        return _IR_FACTORS["gt5y_lt10y"]
    else:
        return _IR_FACTORS["gt10y"]


# ---------------------------------------------------------------------------
# Trade descriptor
# ---------------------------------------------------------------------------

@dataclass
class SACCRTrade:
    """Description of a single trade for SA-CCR purposes.

    Parameters
    ----------
    trade_id : str
    asset_class : str
        One of: "ir", "equity_single", "equity_index", "fx",
        "commodity_energy", "commodity_metals", "commodity_other",
        "credit_ig", "credit_sg".
    notional : float
        Supervisory notional (e.g. swap notional, option notional).
    maturity : float
        Residual maturity in years.
    current_mtm : float
        Current mark-to-market value (from model or market quote).
    delta : float
        Adjusted notional sign/delta: +1 for long/receiver, -1 for short/payer.
        For options, use the Black-Scholes delta.
    start_date : float
        Start of the accrual/coupon period in years (S_j for supervisory
        duration).  Defaults to 0.0 (instrument started today / at issuance).
    end_date : float
        End of the accrual/coupon period in years (E_j for supervisory
        duration).  Defaults to maturity if not explicitly set.
    currency : str
        ISO currency code of the IR hedging set.  Defaults to "USD".
        Only relevant for IR trades; used for cross-currency aggregation.
    mpor : float
        Margin period of risk in years (for margined maturity factor).
        Default is 10 business days (10/252).
    is_margined : bool
        True if the trade is in a margined netting set.
    """
    trade_id: str
    asset_class: str
    notional: float
    maturity: float = 1.0
    current_mtm: float = 0.0
    delta: float = 1.0
    start_date: float = 0.0
    end_date: float = field(default=None)          # type: ignore[assignment]
    currency: str = "USD"
    mpor: float = field(default_factory=lambda: 10 / 252)
    is_margined: bool = False

    def __post_init__(self) -> None:
        # If end_date was not explicitly provided, default it to maturity.
        if self.end_date is None:
            self.end_date = self.maturity


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class SACCRCalculator:
    """SA-CCR EAD calculator for a single netting set.

    Implements the full BCBS 279 / CRE52 formula including Annex B
    supervisory duration and three-bucket IR aggregation.

    Usage::

        calc = SACCRCalculator()
        calc.add_trade(SACCRTrade(
            "swap1", "ir",
            notional=10_000_000,
            maturity=5.0,
            start_date=0.0,
            end_date=5.0,
            current_mtm=50_000,
            currency="USD",
        ))
        calc.add_trade(SACCRTrade(
            "opt1", "equity_single",
            notional=500_000,
            maturity=2.0,
            current_mtm=-10_000,
        ))
        ead = calc.ead()

    The calculator:
      - Applies supervisory duration to IR adjusted notionals (Annex B).
      - Groups IR trades into three maturity buckets per currency.
      - Aggregates across buckets using Annex B correlation (ρ₁₂=ρ₂₃=0.70,
        ρ₁₃=0.30) and sums across currencies.
      - Applies SF_IR = 0.005.
      - Adds non-IR add-ons using the supervisory-factor approach.
      - Computes EAD = alpha × (RC + AddOn).

    Notes
    -----
    Reference: BCBS 279, March 2014 (rev. April 2014), CRE52.
    """

    def __init__(self) -> None:
        self._trades: list[SACCRTrade] = []

    def add_trade(self, trade: SACCRTrade) -> None:
        """Add a trade to the netting set."""
        self._trades.append(trade)

    def replacement_cost(self) -> float:
        """RC = max(V_net, 0) — replacement cost with no collateral."""
        net_mtm = sum(t.current_mtm for t in self._trades)
        return max(net_mtm, 0.0)

    def replacement_cost_with_collateral(self, collateral: float = 0.0) -> float:
        """RC = max(V_net - C, 0) where C is the net collateral held.

        Parameters
        ----------
        collateral : float
            Net collateral held (positive = collateral received).
        """
        net_mtm = sum(t.current_mtm for t in self._trades)
        return max(net_mtm - collateral, 0.0)

    # ------------------------------------------------------------------
    # Full Annex B PFE add-on
    # ------------------------------------------------------------------

    def pfe_addon_full(self) -> float:
        """Full BCBS 279 Annex B PFE add-on across all asset classes.

        IR add-on
        ---------
        1.  For each IR trade j compute:
                d_j  = notional_j × SD(S_j, E_j)       (adjusted notional)
                MF_j = maturity_factor(M_j, ...)
                eff_j = delta_j × d_j × MF_j
        2.  Group by (currency, bucket):
                D_b^ccy = sum_{j in b, ccy} eff_j
        3.  Three-bucket aggregation per currency:
                AddOn_IR^ccy = sqrt(
                    (D1)^2 + (D2)^2 + (D3)^2
                    + rho12*(D1*D2) + rho23*(D2*D3) + rho13*(D1*D3)
                )
        4.  Cross-currency sum (no netting):
                AddOn_IR = sum_ccy AddOn_IR^ccy
        5.  Apply supervisory factor:
                AddOn_IR_final = SF_IR × AddOn_IR

        Non-IR add-on
        -------------
        Simple supervisory-factor approach (unchanged from CRE52.72):
            addon_j = |delta_j| × notional_j × SF_j
        Summed directly (no within-class aggregation).

        Returns
        -------
        float
            Total PFE add-on for the netting set.
        """
        # --- IR: group effective notionals by (currency, bucket) -----------
        # Structure: {currency: {bucket: float}}
        ir_buckets: dict[str, dict[int, float]] = {}

        for trade in self._trades:
            if trade.asset_class != "ir":
                continue

            # Step 1a: supervisory duration adjusted notional
            sd = supervisory_duration(trade.start_date, trade.end_date)
            d_j = trade.notional * sd   # adjusted notional

            # Step 1b: maturity factor
            mf_j = maturity_factor(trade.maturity, trade.is_margined, trade.mpor)

            # Step 1c: signed effective notional contribution
            eff_j = trade.delta * d_j * mf_j

            # Step 2: accumulate into bucket
            ccy = trade.currency
            bkt = _ir_bucket(trade.maturity)

            if ccy not in ir_buckets:
                ir_buckets[ccy] = {1: 0.0, 2: 0.0, 3: 0.0}
            ir_buckets[ccy][bkt] += eff_j

        # --- IR: three-bucket aggregation per currency ---------------------
        addon_ir_total = 0.0

        for ccy, buckets in ir_buckets.items():
            d1 = buckets[1]
            d2 = buckets[2]
            d3 = buckets[3]

            # Annex B three-bucket formula (CRE52 §72):
            #   sqrt( D1^2 + D2^2 + D3^2
            #         + rho12*(D1*D2)
            #         + rho23*(D2*D3)
            #         + rho13*(D1*D3) )
            # Note: the spec writes the cross terms without the factor-of-2
            # that appears in the general quadratic form; the formula as
            # written in BCBS 279 Annex B uses plain products (not 2×).
            variance = (
                d1 ** 2
                + d2 ** 2
                + d3 ** 2
                + _RHO_12 * d1 * d2
                + _RHO_23 * d2 * d3
                + _RHO_13 * d1 * d3
            )
            # Guard against numerical noise producing tiny negatives
            addon_ir_ccy = math.sqrt(max(variance, 0.0))
            addon_ir_total += addon_ir_ccy

        # Step 5: apply IR supervisory factor
        addon_ir_final = _IR_SF * addon_ir_total

        # --- Non-IR: supervisory factor approach ---------------------------
        addon_non_ir = 0.0
        for trade in self._trades:
            if trade.asset_class == "ir":
                continue
            sf = _SF.get(trade.asset_class, 0.0)
            # Effective notional = |delta| × notional × SF
            addon_non_ir += abs(trade.delta) * trade.notional * sf

        return addon_ir_final + addon_non_ir

    def pfe_addon(self) -> float:
        """PFE add-on — delegates to pfe_addon_full() (full Annex B).

        Retained for backward compatibility.
        """
        return self.pfe_addon_full()

    def ead(self) -> float:
        """EAD = alpha × (RC + PFE_add_on).

        Uses the full Annex B add-on and RC with no collateral.
        """
        return _ALPHA * (self.replacement_cost() + self.pfe_addon_full())

    def ead_with_collateral(self, collateral: float = 0.0) -> float:
        """EAD = alpha × (RC_collateral + PFE_add_on).

        Parameters
        ----------
        collateral : float
            Net collateral held (positive = collateral received reduces RC).
        """
        rc = self.replacement_cost_with_collateral(collateral)
        return _ALPHA * (rc + self.pfe_addon_full())

    # ------------------------------------------------------------------
    # Factory helpers: infer SACCRTrade from pipeline Trade objects
    # ------------------------------------------------------------------

    @classmethod
    def from_trades(
        cls,
        trades: list,
        current_mtm: Optional[dict] = None,
    ) -> "SACCRCalculator":
        """Build a SACCRCalculator from a list of Trade objects.

        Parameters
        ----------
        trades : list of Trade
            Trade objects from the pipeline (must have .pricer and .id).
        current_mtm : dict, optional
            Mapping of trade_id -> current MTM value.  Defaults to 0 for
            all trades if not provided.

        Returns
        -------
        SACCRCalculator
        """
        calc = cls()
        mtm_map = current_mtm or {}

        for trade in trades:
            saccr_trade = cls._infer_trade(trade, mtm_map.get(trade.id, 0.0))
            if saccr_trade is not None:
                calc.add_trade(saccr_trade)

        return calc

    @staticmethod
    def _infer_trade(trade, current_mtm: float) -> Optional[SACCRTrade]:
        """Infer SA-CCR attributes from a pipeline Trade object."""
        pricer = trade.pricer
        pricer_type = type(pricer).__name__

        if pricer_type == "InterestRateSwap":
            mat = getattr(pricer, "maturity", 1.0)
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="ir",
                notional=getattr(pricer, "notional", 1_000_000),
                maturity=mat,
                current_mtm=current_mtm,
                delta=1.0 if getattr(pricer, "payer", True) else -1.0,
                # Annex B: start_date = 0 (trade starts today), end_date = maturity
                start_date=0.0,
                end_date=mat,
            )
        elif pricer_type in ("ZeroCouponBond", "FixedRateBond"):
            mat = getattr(pricer, "maturity", 1.0)
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="ir",
                notional=getattr(pricer, "face_value", 1_000_000),
                maturity=mat,
                current_mtm=current_mtm,
                delta=1.0,
                start_date=0.0,
                end_date=mat,
            )
        elif pricer_type in ("EuropeanOption", "BarrierOption", "AsianOption"):
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="equity_single",
                notional=getattr(pricer, "strike", 100.0) * 1_000,  # nominal
                maturity=getattr(pricer, "expiry", 1.0),
                current_mtm=current_mtm,
                delta=1.0,
            )
        elif pricer_type == "GarmanKohlhagen":
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="fx",
                notional=getattr(pricer, "notional", 1_000_000),
                maturity=getattr(pricer, "expiry", 1.0),
                current_mtm=current_mtm,
                delta=1.0,
            )
        else:
            # Unknown pricer type: skip
            return None

    def ead_profile(
        self,
        time_grid: np.ndarray,
        mean_mtm_by_trade: Optional[dict] = None,
    ) -> np.ndarray:
        """Compute SA-CCR EAD at each time step using declining residual maturity.

        At each time ``t`` in ``time_grid``:
        - Each trade's residual maturity is ``max(0, maturity - t)``.
        - IR trades have their start/end dates shifted by ``t``.
        - MTMs are updated from ``mean_mtm_by_trade`` (mean over paths at each step).

        This produces a path-dependent EAD profile suitable for KVA integration:

            KVA = CoC × ∫ EAD(t) dt

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
            Simulation time grid in years.
        mean_mtm_by_trade : dict[str, np.ndarray], optional
            Mapping of ``trade_id`` → ``(T,)`` array of mean MTM at each step.
            If None or a trade is not present, MTM is held at the initial
            ``trade.current_mtm`` for all steps.

        Returns
        -------
        np.ndarray, shape (T,)
            EAD at each time step.
        """
        mtm_map = mean_mtm_by_trade or {}
        T = len(time_grid)
        ead_arr = np.zeros(T)

        for t_idx, t in enumerate(time_grid):
            step_calc = SACCRCalculator()
            for trade in self._trades:
                res_mat = max(0.0, trade.maturity - float(t))
                mtm_series = mtm_map.get(trade.trade_id)
                mtm_t = float(mtm_series[t_idx]) if mtm_series is not None else trade.current_mtm
                updated = SACCRTrade(
                    trade_id=trade.trade_id,
                    asset_class=trade.asset_class,
                    notional=trade.notional,
                    maturity=res_mat,
                    current_mtm=mtm_t,
                    delta=trade.delta,
                    start_date=max(0.0, trade.start_date - float(t)),
                    end_date=max(0.0, trade.end_date - float(t)),
                    currency=trade.currency,
                    mpor=trade.mpor,
                    is_margined=trade.is_margined,
                )
                step_calc.add_trade(updated)
            ead_arr[t_idx] = step_calc.ead()

        return ead_arr

    @staticmethod
    def _supervisory_factor(trade: SACCRTrade) -> float:
        """Return the supervisory factor for a trade.

        For IR trades, returns the single Annex B SF_IR.
        For all other asset classes, returns the tabulated SF.
        """
        if trade.asset_class == "ir":
            return _IR_SF
        return _SF.get(trade.asset_class, 0.0)
