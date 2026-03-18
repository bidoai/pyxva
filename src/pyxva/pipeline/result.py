from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
from typing import Optional
import numpy as np
import pandas as pd

from pyxva.core.market_data import MarketData, ScenarioBump
from pyxva.core.paths import SimulationResult


@dataclass
class NettingSetSummary:
    """Pre-collateral exposure summary for a single netting set."""
    id: str
    ee_profile: np.ndarray    # (T,)
    pfe_profile: np.ndarray   # (T,)
    pse: float
    epe: float


@dataclass
class AgreementResult:
    """
    Full exposure result for one Agreement (ISDA MA + CSA scope).

    Post-collateral profiles are at agreement level.
    Pre-collateral summaries are available per netting set.
    """
    id: str
    counterparty_id: str
    time_grid: np.ndarray

    # Agreement-level profiles (post-collateral)
    ee_profile: np.ndarray
    ene_profile: np.ndarray
    pfe_profile: np.ndarray
    ee_mpor_profile: np.ndarray

    # Per netting set (pre-collateral)
    netting_set_summaries: dict  # str -> NettingSetSummary

    # XVA scalars
    cva: float
    dva: float
    bcva: float
    pse: float
    epe: float
    eepe: float

    # v1.1 xVA fields — default 0.0 (backwards-compatible)
    fva: float = 0.0
    mva: float = 0.0
    kva: float = 0.0

    # Raw MTM arrays — only populated if write_raw_paths=True
    raw_net_mtm: Optional[np.ndarray] = None   # (n_paths, T)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "counterparty_id": self.counterparty_id,
            "cva": self.cva,
            "dva": self.dva,
            "bcva": self.bcva,
            "fva": self.fva,
            "mva": self.mva,
            "kva": self.kva,
            "pse": self.pse,
            "epe": self.epe,
            "eepe": self.eepe,
        }

    def xva_summary(self) -> dict:
        """Return a labelled dict of all xVA scalars for this agreement.

        Includes CVA, DVA, BCVA, FVA, MVA, KVA and the total xVA charge
        (CVA - DVA + FVA + MVA + KVA).
        """
        total = self.cva - self.dva + self.fva + self.mva + self.kva
        return {
            "agreement_id": self.id,
            "counterparty_id": self.counterparty_id,
            "cva": self.cva,
            "dva": self.dva,
            "bcva": self.bcva,
            "fva": self.fva,
            "mva": self.mva,
            "kva": self.kva,
            "total_xva": total,
        }


@dataclass
class RunResult:
    """
    In-memory result of a full RiskEngine pipeline run.

    Holds simulation paths (needed for stress testing) and
    agreement-level summary statistics. Raw per-path MTM arrays
    are not retained unless write_raw_paths=True was set.
    """
    config: object                             # EngineConfig (avoid circular import)
    time_grid: np.ndarray
    simulation_results: dict                   # str -> SimulationResult (kept for stress test)
    agreement_results: dict                    # str -> AgreementResult
    total_cva: float
    total_dva: float
    total_bcva: float
    total_fva: float = 0.0
    total_mva: float = 0.0
    total_kva: float = 0.0

    def summary_df(self) -> pd.DataFrame:
        """One row per agreement with all scalar metrics."""
        rows = []
        for agr_id, result in self.agreement_results.items():
            rows.append(result.to_dict())
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("id")
        return df

    def to_parquet(self, path: str) -> None:
        """
        Write consolidated output:
        - {path}/summary.parquet  — scalar metrics per agreement
        - {path}/ee_profiles.parquet — EE time profile per agreement
        - {path}/pfe_profiles.parquet — PFE time profile per agreement
        If raw_net_mtm is populated on any AgreementResult, also writes
        - {path}/raw_mtm_{agreement_id}.parquet
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Summary scalars
        self.summary_df().to_parquet(os.path.join(path, "summary.parquet"))

        # EE profiles
        ee_data = {
            agr_id: result.ee_profile
            for agr_id, result in self.agreement_results.items()
        }
        pd.DataFrame(ee_data, index=self.time_grid).to_parquet(
            os.path.join(path, "ee_profiles.parquet")
        )

        # PFE profiles
        pfe_data = {
            agr_id: result.pfe_profile
            for agr_id, result in self.agreement_results.items()
        }
        pd.DataFrame(pfe_data, index=self.time_grid).to_parquet(
            os.path.join(path, "pfe_profiles.parquet")
        )

        # Optional raw MTM
        for agr_id, result in self.agreement_results.items():
            if result.raw_net_mtm is not None:
                pd.DataFrame(result.raw_net_mtm, columns=self.time_grid).to_parquet(
                    os.path.join(path, f"raw_mtm_{agr_id}.parquet")
                )

    def to_dict(self) -> dict:
        return {
            "total_cva": self.total_cva,
            "total_dva": self.total_dva,
            "total_bcva": self.total_bcva,
            "total_fva": self.total_fva,
            "total_mva": self.total_mva,
            "total_kva": self.total_kva,
            "agreements": {
                agr_id: result.to_dict()
                for agr_id, result in self.agreement_results.items()
            },
        }

    @classmethod
    def from_parquet(cls, path: str, use_duckdb: bool = False) -> "RunResult":
        """Load a RunResult persisted by ``to_parquet()``.

        Loads scalars eagerly and profile arrays via memory-mapped parquet files.
        The restored RunResult has ``simulation_results = {}`` and
        ``config = None`` (not persisted); stress-testing is not available
        on loaded results.

        Parameters
        ----------
        path : str
            Directory written by ``to_parquet()``.
        use_duckdb : bool
            If True, uses DuckDB to query the parquet files. Requires the
            ``duckdb`` package (optional dependency). When False, uses pandas.

        Returns
        -------
        RunResult
        """
        if use_duckdb:
            try:
                import duckdb
                _load_df = lambda p: duckdb.query(f"SELECT * FROM '{p}'").df()
            except ImportError as exc:
                raise ImportError(
                    "duckdb is required for use_duckdb=True. "
                    "Install it with: pip install duckdb"
                ) from exc
        else:
            _load_df = pd.read_parquet

        # Load scalar summary
        summary_df = _load_df(os.path.join(path, "summary.parquet"))

        # Load profile arrays
        ee_df = _load_df(os.path.join(path, "ee_profiles.parquet"))
        pfe_df = _load_df(os.path.join(path, "pfe_profiles.parquet"))

        time_grid = ee_df.index.to_numpy(dtype=float)

        # Reconstruct AgreementResult objects from persisted data
        agreement_results = {}
        for agr_id in summary_df.index:
            row = summary_df.loc[agr_id]
            # Load optional raw MTM
            raw_path = os.path.join(path, f"raw_mtm_{agr_id}.parquet")
            raw_net_mtm = None
            if os.path.exists(raw_path):
                raw_df = _load_df(raw_path)
                raw_net_mtm = raw_df.to_numpy(dtype=float)

            agreement_results[agr_id] = AgreementResult(
                id=agr_id,
                counterparty_id=str(row.get("counterparty_id", "")),
                time_grid=time_grid,
                ee_profile=ee_df[agr_id].to_numpy(dtype=float),
                ene_profile=np.zeros_like(time_grid),  # not persisted
                pfe_profile=pfe_df[agr_id].to_numpy(dtype=float),
                ee_mpor_profile=np.zeros_like(time_grid),  # not persisted
                netting_set_summaries={},
                cva=float(row.get("cva", 0.0)),
                dva=float(row.get("dva", 0.0)),
                bcva=float(row.get("bcva", 0.0)),
                fva=float(row.get("fva", 0.0)),
                mva=float(row.get("mva", 0.0)),
                kva=float(row.get("kva", 0.0)),
                pse=float(row.get("pse", 0.0)),
                epe=float(row.get("epe", 0.0)),
                eepe=float(row.get("eepe", 0.0)),
                raw_net_mtm=raw_net_mtm,
            )

        # Aggregate totals from agreement-level values
        total_cva = sum(r.cva for r in agreement_results.values())
        total_dva = sum(r.dva for r in agreement_results.values())
        total_bcva = sum(r.bcva for r in agreement_results.values())
        total_fva = sum(r.fva for r in agreement_results.values())
        total_mva = sum(r.mva for r in agreement_results.values())
        total_kva = sum(r.kva for r in agreement_results.values())

        return cls(
            config=None,
            time_grid=time_grid,
            simulation_results={},
            agreement_results=agreement_results,
            total_cva=total_cva,
            total_dva=total_dva,
            total_bcva=total_bcva,
            total_fva=total_fva,
            total_mva=total_mva,
            total_kva=total_kva,
        )

    def stress_test(
        self,
        bumps: list,
        market_data: MarketData,
    ) -> "RunResult":
        """
        Reprice on existing simulation paths with bumped market data.
        No re-simulation. Returns a new RunResult.

        bumps: list[ScenarioBump]
        market_data: the base MarketData to apply bumps to
        """
        bumped_md = market_data.scenario(bumps)
        # Re-run phases 2+3 with the bumped market data but same simulation_results
        from pyxva.pipeline.engine import RiskEngine
        return RiskEngine._run_exposure_phase(
            config=self.config,
            time_grid=self.time_grid,
            simulation_results=self.simulation_results,
            market_data=bumped_md,
        )
