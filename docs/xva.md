# XVA: A Practitioner's Guide

> Written for a CCR quant with margin desk experience who wants to understand the full XVA picture — how the adjustments are derived, how desks operate, how hedging works, and where the bodies are buried.

---

## Table of Contents

1. [The Problem XVA Solves](#1-the-problem-xva-solves)
2. [The Pre-Crisis World vs Post-Crisis Reality](#2-the-pre-crisis-world-vs-post-crisis-reality)
3. [CSA and ISDA Mechanics — In Depth](#3-csa-and-isda-mechanics--in-depth)
4. [Exposure Fundamentals](#4-exposure-fundamentals)
5. [CVA — Credit Valuation Adjustment](#5-cva--credit-valuation-adjustment)
6. [DVA — Debt Valuation Adjustment](#6-dva--debt-valuation-adjustment)
7. [FVA — Funding Valuation Adjustment](#7-fva--funding-valuation-adjustment)
8. [MVA — Margin Valuation Adjustment](#8-mva--margin-valuation-adjustment)
9. [KVA — Capital Valuation Adjustment](#9-kva--capital-valuation-adjustment)
10. [ColVA and Other Adjustments](#10-colva-and-other-adjustments)
11. [The XVA Interaction Problem](#11-the-xva-interaction-problem)
12. [How XVA Desks Operate](#12-how-xva-desks-operate)
13. [Hedging XVA](#13-hedging-xva)
14. [Regulatory Framework](#14-regulatory-framework)
15. [Internal Transfer Pricing](#15-internal-transfer-pricing)
16. [Technology and Quant Infrastructure](#16-technology-and-quant-infrastructure)
17. [Open Debates and Unsolved Problems](#17-open-debates-and-unsolved-problems)

---

## 1. The Problem XVA Solves

The textbook price of an OTC derivative assumes the trade exists in a frictionless world: counterparties never default, funding is free, capital is costless, and collateral earns the risk-free rate. None of these assumptions hold in practice.

XVA — the collective term for valuation adjustments — converts a theoretical mid-market price into the true economic cost of holding a trade, accounting for:

- **Counterparty default risk** (CVA)
- **Own default benefit** (DVA) and the moral hazard this creates
- **The cost of funding uncollateralised positions** (FVA)
- **The cost of posting initial margin** (MVA)
- **The cost of regulatory capital** (KVA)
- **The optionality embedded in collateral agreements** (ColVA)

From your perspective on a margin desk, you were already computing EE, EPE, and CSB profiles — XVA is the process of pricing those profiles and charging them to the business.

The fundamental insight: **a derivative is not just a stream of cashflows. It is also a credit facility, a funding commitment, a capital consumer, and a collateral agreement — and each of those components has a cost that should be priced.**

---

## 2. The Pre-Crisis World vs Post-Crisis Reality

### Before 2008

OTC derivatives were priced assuming:

1. All counterparties were effectively risk-free (interbank market)
2. LIBOR was the universal funding rate (and could be used freely)
3. Collateral postings were negligible — many trades ran uncollateralised
4. Capital was cheap; banks held thin buffers

CVA existed in principle but was treated as a small residual adjustment, often handled by a back-office reserve rather than actively priced. FVA, MVA, and KVA were not recognised concepts at all.

### The Crisis Inflection Points

**2008–2009: Lehman / Bear Stearns.** The assumption that large dealer banks could not default was shattered. CVA losses dwarfed mark-to-model losses for many firms. Basel III's CVA capital charge was born from the observation that roughly two-thirds of crisis-era credit losses came from CVA mark-to-market moves, not actual defaults.

**2010–2012: FVA debate erupts.** Major dealers (Goldman, JPMorgan) began charging FVA on uncollateralised trades. The argument: the bank must fund the initial MtM of the trade in the wholesale market, paying a spread over OFR (Overnight Funding Rate). That spread is a real cost. The academic community (Hull, White) pushed back, arguing FVA creates double-counting with DVA. The debate was never fully resolved; practitioners won by default.

**2015–2020: IM mandate.** BCBS-IOSCO phased in mandatory initial margin for bilateral non-cleared OTC derivatives. Phase 1 (largest dealers, 2016) through Phase 6 (smaller buy-side, 2022). MVA — the cost of funding IM — went from a theoretical concept to a six-figure charge on a vanilla IRS.

**Post-2020: KVA maturity.** The Basel III / Basel IV capital framework (SA-CCR replacing CEM, FRTB for market risk) made capital costs quantifiable and forward-looking enough to price. KVA became a real P&L line on some desks.

---

## 3. CSA and ISDA Mechanics — In Depth

You lived this on the margin desk, but let's establish the full picture for the XVA context.

### The Document Hierarchy

```
ISDA Master Agreement (1992 or 2002)
  └─ Schedule (bilateral elections, netting scope)
       └─ Credit Support Annex (CSA)
            ├─ Variation Margin (VM) terms
            └─ Initial Margin (IM) terms  [separate 2018 IM CSA]
```

**ISDA Master Agreement**: Governs close-out netting. In default, all trades under the agreement are terminated simultaneously and netted to a single payment obligation. This is legally enforced netting — the legal opinions your firm obtained in each jurisdiction were expensive precisely because this netting benefit is massive in capital terms.

**Schedule**: Bilateral elections. Particularly important: the definition of "Affected Party" (one-way CSA vs two-way), the list of Termination Events, and the set of Eligible Credit Support.

**Credit Support Annex**: The margin agreement. Two generations matter:

- **1994 NY law / 1995 English law CSA**: The legacy document. Title transfer (English) vs pledge (NY). Under title transfer, posted collateral is re-hypothecatable — the receiver legally owns it. This creates ColVA (see below). Under pledge, the poster retains ownership.
- **2016 VM CSA**: The mandatory regulatory document for Phase 1+ counterparties. Cash-only for regulatory VM (in most jurisdictions). Daily margining. Essentially eliminates threshold and MTA for in-scope counterparties. No re-hypothecation complexity for cash.
- **2018 IM CSA / SCSA**: Governs IM posting to third-party custodians (triparty: BNY Mellon, Euroclear). IM is segregated — the receiver cannot re-use it. This is critical: MVA is a real cost precisely because the IM earns nothing above the custodian rate.

### Key CSA Parameters and Their XVA Impact

| Parameter | Symbol | XVA Effect |
|---|---|---|
| Threshold (counterparty) | TH_c | Raises uncollateralised EE by TH_c; directly increases CVA |
| Threshold (own) | TH_p | Raises ENE; increases DVA |
| MTA (party) | MTA_p | Creates gap risk — margin call may be too small to trigger |
| MTA (counterparty) | MTA_c | Symmetric for DVA |
| MPOR / Cure period | τ | Gap between last good margin call and default; EE shifts forward by τ |
| Margin call frequency | f | Daily calls → near-zero gap; weekly → up to 5-day gap |
| Initial Margin (IM) | IM | Directly reduces collateralised EE; drives MVA |
| IA (Independent Amount) | IA | Like a pre-posted IM; reduces EE at inception |
| Eligible collateral | — | Drives ColVA (cash vs bonds, haircuts, currency mismatch) |
| Rounding | r | Small effect; rounds CSB to nearest unit |

### MPOR in Detail

MPOR (Margin Period of Risk) is the time window over which the bank is exposed after a counterparty default. Under Basel and ISDA's assumptions:

```
MPOR = time to detect default + time to stop accepting calls + cure period + liquidation
```

The regulatory minimum is:
- **10 business days** for bilateral non-cleared OTC (vanilla)
- **20 business days** if the netting set has >5000 trades
- **5 business days** for ETDs and centrally cleared

For XVA purposes, MPOR determines how far forward you shift the last-good CSB to compute collateralised exposure:

```
E_coll(t) = max(V(t) − CSB(t − τ) − IM(t), 0)
```

where τ is MPOR. This is what your streaming engine's MPOR ring buffer was computing.

### The Gap Risk Problem

Even with daily margining, there is residual exposure between the last successful margin call and default. This gap exposure is what makes MPOR non-trivial:

```
Gap exposure = V(t) − V(t − τ)  [approximately]
```

For a 10-day MPOR and typical IRS volatility of ~50bps/day, the gap exposure can easily be 5–15% of notional on a long-dated swap. This is the residual CVA you pay even on a fully-collateralised agreement.

### One-Way vs Two-Way CSAs

**One-way CSA**: Only one party posts collateral (typically a supranational or sovereign that refuses to post). The bank posts VM to the client but receives nothing back. From the bank's perspective: full DVA benefit (they can always default to zero), zero CVA protection. This is genuinely bad asymmetric exposure — the bank is funding the client's potential gain for free.

**Asymmetric threshold CSAs**: The client has a threshold (say, £5m) below which they don't post. The bank has TH = 0. Exposure is floored at −TH_c for CVA purposes. Dealers price a CVA uplift for this structure.

---

## 4. Exposure Fundamentals

You know these from the margin desk, but let's unify the notation for XVA derivations.

### Definitions

Let V(t) = net MtM of the netting set from the bank's perspective at time t.

| Metric | Formula | Usage |
|---|---|---|
| EE(t) | E[max(V(t), 0)] | CVA, FVA (lending leg) |
| ENE(t) | E[min(V(t), 0)] | DVA, FVA (borrowing leg) |
| EPE | (1/T) ∫₀ᵀ EE(t) dt | Capital (SA-CCR EAD proxy) |
| EEPE | Effective EPE, monotone 1yr average | IMM capital |
| PFE(α, t) | quantile_α(max(V(t), 0)) | Limit management |
| CSB(t) | Collateral balance (from path_csb) | Collateralised exposure |

### Uncollateralised vs Collateralised Exposure

**Uncollateralised**: EE(t) is computed directly from simulated paths. Used for CVA when TH = ∞ (no CSA).

**Partially collateralised**: When TH > 0 or MTA > 0:
```
EE_coll(t) = E[max(V(t) − CSB_path(t − MPOR), 0)]
```
The CSB depends on the entire path history (MTA gating), which is why `path_csb` in the code must be computed path-by-path (the Numba kernel).

**Fully collateralised with IM**:
```
EE_coll(t) = E[max(V(t) − CSB_path(t − MPOR) − IM(t), 0)]
```
With large IM (SIMM-based IM is typically 10–15% of notional for IR swaps), this can be near zero except for the gap risk.

### The Effective EPE vs Standard EPE Distinction

For regulatory IMM capital, Basel requires EEPE (Effective Expected Positive Exposure):

```
EE_eff(t) = max(EE(t), EE_eff(t_prev))   [monotone EE]
EEPE = (1/1yr) ∫₀¹ EE_eff(t) dt
```

The monotonicity requirement is anti-gaming — it prevents cherry-picking a time grid where EE happens to be low. For CVA P&L purposes you use standard EPE, but for capital you use EEPE.

---

## 5. CVA — Credit Valuation Adjustment

### Economic Meaning

CVA is the price of counterparty default risk embedded in an OTC derivative. It is the expected loss from counterparty default, discounted to today.

### The Formula

$$\text{CVA} = -\text{LGD} \sum_{i=1}^{T} \text{EE}(t_i) \cdot [Q(t_{i-1}) - Q(t_i)]$$

where:
- LGD = Loss Given Default (1 − Recovery Rate, typically 0.4–0.6)
- EE(t_i) = discounted Expected Exposure at time t_i
- Q(t) = risk-neutral survival probability to t, derived from CDS spreads

The sum is over time buckets covering the life of the portfolio.

### Derivation Intuition

Think of each time bucket [t_{i-1}, t_i] as an independent scenario:
- With probability Q(t_{i-1}) − Q(t_i), the counterparty defaults in that window
- If they do, you lose EE(t_i) (your in-the-money exposure at that moment)
- Your actual recovery is (1 − LGD) of that exposure

This is the credit leg of a CDS, where the reference obligation is your OTC portfolio rather than a bond.

### CVA as a Contingent CDS

The more precise view: CVA = value of a CDS-like instrument where:
- Protection notional = V(t) at the time of default
- Premium = spread embedded in the price of the original trade

This is why CVA desks hedge using single-name CDS and CDS indices — they are replicating the protection buyer position.

### Bilateral vs Unilateral CVA

**Unilateral CVA**: Only the counterparty can default. Bank is assumed risk-free. Simple but ignores own credit.

**Bilateral CVA (BCVA)**: Bank + counterparty can both default. BCVA = CVA − DVA (see section 6). More theoretically complete; creates practical problems.

### CDS-Implied vs Historical Hazard Rates

**Risk-neutral (CDS-implied)**: Use the hazard rate bootstrapped from CDS spreads. Prices the market cost of hedging. Correct for P&L and hedging. The problem: CDS markets exist for maybe 500 names; you have 5000 counterparties. The rest need proxy curves.

**Proxy methodologies**:
- Sector/rating mapped curves: take the CDS index for BB energy, apply to an unrated energy company
- Bond-implied spreads: bootstrap from asset swap spreads where available
- Internal credit ratings: map to an internal PD curve from the bank's credit model

**Historical (actuarial)**: Use historical default rates by rating bucket. Correct for IFRS 9 provisioning (ECL), not for hedge-aligned CVA P&L.

### CVA Greeks and Risk Management

The main risk dimensions of CVA:

| Risk | Driver | Hedge |
|---|---|---|
| CS01 | CDS spread move on counterparty | Single-name CDS |
| CS01 (index) | Macro credit spread widening | iTraxx / CDX |
| IR01 | Rate move changes EE profile | Swaptions, IR delta from underlying |
| Delta/Vega | Underlying moves change MtM | Options, variance swaps |
| Wrong-Way Risk | Correlation between exposure and default | Difficult; bespoke structures |

### Wrong-Way Risk

**General WWR**: Weak positive correlation between exposure and counterparty PD. Example: FX forwards with an EM counterparty — if their currency depreciates (you gain on the forward), it's often because their economy is in stress (PD rises). Your EE and their default probability are positively correlated.

**Specific WWR**: Strong structural link. Example: the counterparty has sold you a put on their own stock. If they default, their stock is likely at zero — the put is deep in the money exactly when they default.

ISDA's alpha multiplier (1.4× applied to EAD in IMM models) is partly a regulatory penalty for not modelling WWR correctly.

### IFRS 13 and Fair Value CVA

Under IFRS 13, derivatives must be measured at fair value, which includes CVA. This is not optional for reporting entities. The Basel CVA capital charge is separate — it covers the risk of CVA mark-to-market volatility, not just expected loss.

---

## 6. DVA — Debt Valuation Adjustment

### The Uncomfortable Truth

DVA is the value to the bank of the possibility that the **bank itself** defaults. If the bank goes bust and owes money on a derivative, it will pay less than face value. Therefore, today's fair value should reflect that discount.

$$\text{DVA} = \text{LGD}_{\text{own}} \sum_{i=1}^{T} |\text{ENE}(t_i)| \cdot [Q_{\text{own}}(t_{i-1}) - Q_{\text{own}}(t_i)]$$

### Why DVA is Controversial

**The monetisation problem**: DVA increases as the bank's own credit deteriorates. A bank in financial stress books DVA gains. Lehman booked DVA gains in Q3 2008 as its own spreads widened. This is perverse — you are recognising profit from your own impending default.

**The hedging impossibility**: To hedge DVA you would need to sell protection on your own default — i.e., sell CDS on yourself. Banks cannot do this in any meaningful size. Some tried buying back their own debt or shorting financial indices, but the hedge is imperfect at best.

**The accounting asymmetry**: Under IFRS 13, you must include DVA in fair value. Under Basel III, DVA cannot offset CVA for capital purposes. So you get the P&L credit but still hold capital against the gross CVA.

### The BCVA Decomposition

$$\text{BCVA} = \text{CVA} - \text{DVA}$$

In a fully bilateral world where neither party is risk-free:
- When your counterparty is in the money: you're the credit risk taker → CVA applies
- When you're in the money to the counterparty: they're the credit risk taker → DVA applies

The sum = the bilateral fair value of default-risky derivatives. In principle, both parties should agree on BCVA (pricing in each other's credits). In practice they don't, creating a bid-ask spread and negotiation friction.

### DVA and FVA: The Connection

This is where it gets subtle. As the bank's credit spread widens:
1. DVA increases (booking a gain)
2. The bank's unsecured funding cost also increases

The funding cost increase is the economic reality; DVA is the accounting recognition. Hull & White (2012) argued FVA is double-counting DVA — the funding cost is already captured in DVA. Most practitioners disagree: DVA is a derivative fair value adjustment; FVA is a real treasury funding cost. They inhabit different parts of the P&L and balance sheet.

---

## 7. FVA — Funding Valuation Adjustment

### The Funding Problem

Consider an uncollateralised interest rate swap with a corporate counterparty. The bank executes at mid-market. Then it hedges in the inter-dealer market under a standard CSA (daily cash VM, zero threshold).

The hedge is collateralised but the original trade is not. When the swap moves in the bank's favour:
- The corporate owes the bank MtM (but posts no collateral)
- The bank owes its hedge counterparty VM cash under the CSA

The bank must fund that VM cash in the wholesale market. If LIBOR/SOFR + 80bps is the bank's funding cost and OIS is the risk-free rate, the bank is paying 80bps on the funded amount for the life of the positive exposure.

This funding cost — not captured in the risk-free derivative price — is FVA.

### The Formula

$$\text{FVA} = -\text{Funding Spread} \int_0^T \text{EE}(t) \, dt + \text{Funding Spread} \int_0^T |\text{ENE}(t)| \, dt$$

The first term is the cost of funding your positive exposure (lending leg). The second is the benefit of receiving funding from the counterparty's negative exposure (borrowing leg).

Or equivalently:

$$\text{FVA} = s_f \int_0^T [\text{EE}(t) - |\text{ENE}(t)|] \, dt$$

where s_f is the funding spread. Note this is the net funding cost — when you're in the money you're a net borrower; when you're out of the money you're a net lender.

### FCA vs FBA

Some firms split FVA into two components:

**FCA (Funding Cost Adjustment)**: Cost of funding positive exposure. Always a charge.
```
FCA = s_f × EPE
```

**FBA (Funding Benefit Adjustment)**: Benefit from the counterparty implicitly funding your negative MtM. A credit.
```
FBA = s_f × |ENE|
```

The split matters for internal allocation (see section 15) and for acknowledging the asymmetry between the bank's borrowing cost and lending rate.

### The Discount Rate Debate

**OIS discounting** became standard post-crisis. Under OIS discounting, fully-collateralised derivatives are priced at risk-free. FVA then arises as the residual funding cost for uncollateralised MtM.

The discount rate used in FVA calculations should be the **marginal cost of unsecured funding** — typically the bank's internal funding rate (IFR) as set by treasury. This rate:
- Is above OIS by the bank's credit spread
- Varies by tenor and currency
- Changes as the bank's funding conditions change

### FVA Asymmetry: The Dealer Dilemma

If counterparty A and counterparty B both have uncollateralised CSAs with the same dealer, and they are perfect offsets:
- The dealer has zero net market risk
- But they pay FCA on EE with A and receive FBA on ENE with B
- FCA ≠ FBA if the bank's borrowing and lending rates differ (typical)

This is the "funding asymmetry" — banks are not funding-neutral intermediaries. They borrow short, lend long, and have an asymmetric cost structure.

---

## 8. MVA — Margin Valuation Adjustment

### The IM Problem

Initial margin under BCBS-IOSCO rules is designed to cover closeout costs over MPOR. For a 5-year IRS, SIMM IM is roughly:

```
IM ≈ Sensitivity × RiskWeight × MPOR scaling
```

For a €100m notional 5yr EUR swap (DV01 ≈ €45k/bp), at SIMM RW of ~48bps:
```
IM ≈ €45k × 480 × sqrt(10/250) = ~€1.37m
```

This IM is posted to a segregated custodian and earns approximately risk-free rate (SOFR/ESTR). The bank funds it at SOFR + its credit spread. The cost of that spread over the life of the trade is MVA.

### The Formula

$$\text{MVA} = s_f \int_0^T \text{E}[\text{IM}(t)] \, dt$$

where:
- s_f = bank's marginal funding spread (same as FVA)
- E[IM(t)] = expected IM profile under the risk-neutral measure

### SIMM in the XVA Context

SIMM (Standard Initial Margin Model) is path-dependent: it depends on the portfolio sensitivities at each future simulation date. Computing E[IM(t)] on a simulation path requires either:

1. **Regression-based IM**: At each path and time step, regress IM against state variables (rates, spot, vol). Computationally expensive.
2. **SIMM approximation**: Approximate the future sensitivity as a function of the future MtM (delta-based proxy).
3. **Schedule IM**: Use the regulatory schedule method (% of notional). Crude but tractable.

The MVA literature (Green, Kenyon) uses approach 1 as the gold standard but practitioners typically use approach 2 or 3 for speed.

### MVA and Trade Netting

MVA exhibits strong netting effects. Adding an offsetting trade to a netting set can significantly reduce IM and therefore MVA. This is a key lever in XVA desk trading — a new trade might have low standalone MVA if it offsets existing exposure.

This is also the driver of compression services (TriOptima, Quantile): compressing offsetting trades reduces gross IM, which reduces MVA for the entire portfolio.

### MVA as a Competitive Disadvantage

Post Phase 5/6 rollout, smaller buy-side firms faced significant MVA on standard hedges. A vanilla 5yr IRS that once had near-zero XVA now carries:
- CVA: £20k–£100k depending on credit quality and CSA terms
- MVA: £30k–£150k depending on portfolio netting and IM model
- FVA: £10k–£50k for partially-collateralised structures

This is why cleared derivatives became attractive — CCP IM is netted across all clearing members, and cleared VM is daily cash, eliminating gap risk. The clearing mandate exists partly to eliminate bilateral XVA complexity.

---

## 9. KVA — Capital Valuation Adjustment

### What Capital Costs

KVA prices the ongoing regulatory capital consumed by a trade over its life. The logic: the bank must maintain CET1 capital against derivative exposures under Basel III. That capital has a cost (shareholders require a return above the risk-free rate). That cost should be attributed to the trades that consume the capital.

### The Formula

$$\text{KVA} = \text{CoC} \int_0^T E[\text{RWA}(t)] \cdot \frac{8\%}{12.5} \, dt$$

More simply:

$$\text{KVA} = \text{CoC} \times \text{EAD}(t_0) \times T \quad \text{(flat approximation)}$$

where:
- CoC = cost of capital (return on equity target minus risk-free rate), typically 8–15%
- EAD = Exposure at Default under SA-CCR or IMM
- RWA = risk-weighted assets = 12.5 × capital requirement

The flat approximation (using today's EAD) is crude. The full model requires simulating future capital requirements, which means simulating future SA-CCR/IMM EAD — a significant computational challenge.

### SA-CCR and Its Effect on KVA

SA-CCR (Standard Approach for Counterparty Credit Risk, CRE52) replaced CEM in 2017–2019 and is now the standard for banks not approved for IMM. SA-CCR EAD formula:

```
EAD = alpha × (RC + PFE_aggregate)
alpha = 1.4
RC = Replacement Cost = max(V - C, 0)  [collateralised netting sets]
PFE = multiplier × AddOn
```

The AddOn is computed using SACCR supervisory factors by asset class — IR, FX, Credit, Equity, Commodity. These factors are deliberately conservative to create incentives for clearing and collateralisation.

SA-CCR KVA is more tractable than IMM KVA because SA-CCR can be approximated analytically (no simulation needed). IMM KVA requires simulating the future internal model EAD, which is model-within-model.

### CVA Capital: The VaR on Top of CVA

Beyond the capital for default risk, Basel III introduced a separate **CVA capital charge** for the market risk of CVA P&L fluctuations. There are two approaches:

**SA-CVA** (FRTB-CVA): Sensitivity-based. Computes delta and vega sensitivities of CVA to market risk factors, then applies supervisory correlations. Requires a CVA risk model that maps hedges to approved instruments.

**BA-CVA** (Basic Approach): Formula-based. No internal model. No capital reduction for hedges except for eligible CDS on the counterparty. Punitive for unhedged CVA books.

KVA should incorporate the CVA capital charge, not just the default-risk capital. Many KVA implementations ignore this complexity; sophisticated desks don't.

---

## 10. ColVA and Other Adjustments

### ColVA (Collateral Valuation Adjustment)

Under legacy 1994/1995 CSAs with title transfer and non-cash collateral, the receiver of collateral can re-use it (re-hypothecation). This creates optionality:

1. **Currency optionality**: The CSA may permit posting in multiple currencies. The poster will deliver the cheapest-to-deliver currency. The receiver pays OIS on what they receive but gets the CDS-cheapest-to-deliver from the deliverer. The value of this option is ColVA.

2. **Bond haircut optionality**: The poster chooses which eligible bonds to deliver. Low-yielding bonds may have low opportunity cost, making them optimal to deliver. The receiver's return on this collateral is the bond coupon net of haircut — potentially different from OIS.

3. **Re-hypothecation benefit**: If the receiver can re-use posted collateral (pledge it as their own VM to a third party), they earn a re-hypothecation benefit. This disappears under the 2016 VM CSA and 2018 IM CSA for in-scope counterparties.

ColVA is computed as:

```
ColVA = E[CSB × (OIS - actual_collateral_rate)] over all paths
```

For legacy books with mixed collateral, this can be material (50–200bps on notional for long-dated cross-currency swaps with optionality).

### OIS-LIBOR Basis (Historical Context)

Pre-2022, before LIBOR cessation, the choice of discount rate for collateralised vs uncollateralised trades was a ColVA-type adjustment. Trades collateralised in USD cash were discounted at SOFR/Fed Funds; EUR trades at ESTR. The OIS-LIBOR basis (typically 5–30bps) was itself a ColVA. Post-LIBOR transition this is less relevant but the multi-curve framework persists.

### XVA from the CCPs' Perspective

Central counterparty clearing replaces bilateral CVA with:
- **Initial Margin** to the CCP (drives MVA for clearing members)
- **Variation Margin** in cash daily
- **Default fund contributions** (a form of KVA — you are capitalising the CCP's default fund)

The "clearing XVA" is therefore dominated by MVA + KVA on default fund. For vanilla IRS and CDS, cleared XVA is typically lower than bilateral XVA because of IM netting across the entire cleared book.

---

## 11. The XVA Interaction Problem

XVA adjustments are not additive. They interact, and naive summation double-counts.

### The Overlap

**CVA and FVA**: Both depend on EE. CVA is the cost of losing money when the counterparty defaults; FVA is the cost of funding that exposure until they pay or default. They are not the same thing but they are correlated — large positive EE drives both.

**DVA and FVA**: Hull & White's original critique: DVA already compensates for the bank's funding disadvantage. In the limit where the bank's credit spread = its funding spread, DVA = FBA exactly. In practice, they diverge because DVA is measured risk-neutrally against CDS spread while FBA uses the actual treasury funding rate.

**CVA and KVA**: Both depend on default probability. CVA is the expected loss; KVA includes capital against unexpected loss (the VaR/ES component). They should not be added without adjusting for the capital already implicitly charged in CVA.

**MVA and KVA**: IM reduces CVA (by reducing EE). But IM costs MVA. And the capital treatment of cleared trades (lower SA-CCR EAD) reduces KVA. The optimal collateral strategy minimises MVA + KVA jointly — not each independently.

### A Unified Framework

The most rigorous treatment (Green, Kenyon, Lichters 2015 / Albanese, Andersen 2014) computes XVA in a consistent framework where:

1. All adjustments come from a single pricing PDE / simulation
2. Close-out is handled consistently (RFC vs CPC: risk-free close-out vs market close-out)
3. Capital and margin requirements are path-dependent inputs

In practice, most desks run approximate additive models because the unified framework is computationally prohibitive. The approximation error is accepted as a model risk.

---

## 12. How XVA Desks Operate

### Organisational Models

There are three main organisational structures:

**1. Centralised XVA Desk (Most Common at G-SIBs)**

A single desk (sometimes split into CVA desk + FVA/MVA desk) holds the XVA reserves and manages the Greeks centrally. Trading desks transact at "clean" prices (pre-XVA) and pay an XVA charge to the central desk at inception. The XVA desk manages the residual risk.

Advantages: economies of scale in hedging, consistent pricing, capital efficiency.
Disadvantages: internal transfer pricing disputes, distance from business.

**2. Distributed XVA (Embedded in Trading Desks)**

Each trading desk (rates, credit, equity) manages its own XVA. More common at smaller dealers or where product complexity is high enough that XVA is deeply product-specific.

Advantages: better product knowledge, direct P&L accountability.
Disadvantages: duplication of infrastructure, no cross-product netting of hedges.

**3. Risk Management Only (No P&L)**

XVA is computed for regulatory and accounting purposes but not charged to the business — it sits in a central reserve. Common at less sophisticated firms. The XVA is marked but not actively hedged.

### The Day-to-Day of an XVA Desk

**New Trade Pricing**: When a sales desk wants to execute a trade, the XVA desk provides a charge. This happens in seconds for vanilla flow, minutes for structured. The charge is the sum of CVA + FVA + MVA + KVA for the incremental trade in the netting set.

Incremental pricing is highly non-linear: a trade that offsets existing exposure might have negative CVA (it reduces the netting set's EE). A trade that increases exposure may have high CVA plus netting charges.

**Greeks Monitoring**: The XVA desk runs a book of CVA Greeks (CS01 by counterparty, aggregate IR01, vega). These are the hedges needed to maintain a delta-neutral CVA book. The Greeks are typically re-run overnight; intraday approximations are used for flow.

**Collateral Desk Interface**: XVA desks work closely with the collateral/margin team to understand:
- Which CSAs are active vs legacy
- MPOR estimates by netting set
- IM call schedules and custodian balances
- Disputes and fails (which create gap exposure above the model)

**XVA P&L Attribution**: Daily P&L is explained by:
- New trade CVA charges (inception day 1 P&L)
- CVA mark-to-market: spread moves, rate moves, vol moves
- Theta (time decay of CVA)
- Defaults and credit events (rarely, but when they happen — large)
- FVA carry: actual funding cost vs model
- MVA P&L: IM movements × funding spread

**The CVA Book Structure**: The CVA book is effectively a portfolio of credit-linked products. Its P&L looks like a CDS portfolio:
- Long credit risk: the bank is the protection seller on its counterparty exposures
- Short credit risk (if hedging with CDS): the bank buys protection on the counterparty

A fully-hedged CVA book should have near-zero P&L from spread moves. An unhedged book has large CS01 exposure.

---

## 13. Hedging XVA

### CVA Hedging

**What to hedge**: The main delta is CS01 — sensitivity of CVA to the counterparty's CDS spread. For a netting set with EPE = €10m and 5yr maturity, at 100bps spread:

```
CS01 ≈ EPE × tenor_duration ≈ €10m × 4yr / 10000 ≈ €4,000/bp
```

To hedge, buy €4k/bp of CDS protection on the counterparty. As spreads widen, CVA increases (mark-to-market loss) but the CDS hedge gains by an equal amount.

**Cross-gamma**: The CS01 itself moves when rates or the underlying moves (because EE moves). This cross-gamma (∂²CVA/∂spread∂rate) is difficult to hedge — it requires options on credit + rates simultaneously (credit swaptions, callable bonds).

**Index hedges**: For counterparties without liquid single-name CDS, use iTraxx or CDX sector baskets. The hedge is imperfect (basis risk) but reduces aggregate credit spread sensitivity. Typical basis risk: 30–50% of notional hedge effectiveness.

**Impossible to hedge**:
- Jump-to-default (JTD): the CVA loss when the counterparty unexpectedly defaults
- Wrong-way risk: correlation between exposure and credit quality
- Model risk: the spread or EE model being wrong

### FVA and MVA Hedging

**FVA is not hedgeable in the traditional sense.** The exposure is funded in the wholesale market — the hedge is simply ensuring you have the funding. The FVA charge at trade inception is the present value of future funding costs; whether you actually fund at that spread or better/worse is a funding execution question, not a hedge.

Some desks view FVA as "earning back" the FCA: if you charge the client FVA upfront, you earn the funding cost spread over the trade's life if funding conditions stay constant. The risk is funding spread moves (your IFR widens/tightens).

**MVA hedging**: Even less hedgeable. IM changes as portfolio Greeks change — a rate move might increase SIMM IM by 10%. You cannot easily hedge the convexity of IM with respect to the risk factors. Some desks take SIMM vega (IM sensitivity to vol) as a book Greek.

### The CVA Hedge Accounting Dilemma

Under IFRS 9, CVA hedges can only reduce P&L volatility if they qualify for hedge accounting. CDS hedges on counterparty credit often fail hedge accounting tests because:
1. The hedged item (CVA) is not a separately designated financial instrument
2. The hedge effectiveness requirements are strict

Many banks take P&L volatility from CVA but get capital relief from CVA hedges (SA-CVA allows partial capital reduction for approved hedges). The accounting and capital treatments are misaligned.

---

## 14. Regulatory Framework

### The Timeline

| Year | Rule | Impact |
|---|---|---|
| 2013 | Basel III / CRD4 live | CVA capital charge (standardised) |
| 2014 | IMM CVA advanced approach | VaR-based CVA capital for large banks |
| 2017 | SA-CCR (CRE52) | Replaced CEM for CCR EAD |
| 2019 | FRTB-CVA consultation | SA-CVA and BA-CVA |
| 2016–2022 | BCBS-IOSCO IM phases 1–6 | Mandatory bilateral IM |
| 2023–2025 | Basel IV / CRR3 | SA-CVA mandatory, IMM-CVA more prescriptive |

### SA-CCR (CRE52) — What Quants Need to Know

SA-CCR computes EAD as:

```
EAD = alpha × (RC + PFE_SACCR)
alpha = 1.4
```

**RC (Replacement Cost)**:
- Unmargined: RC = max(V_netting_set − C, 0)
- Margined: RC = max(V − C_VM − C_IM, TH + MTA − NICA, 0)

where NICA = net independent collateral amount (IM received net of IM posted).

**PFE_SACCR**:
```
PFE = multiplier × AddOn_aggregate
multiplier = min(1, 0.05 + 0.95 × exp(V − C) / (2 × 0.95 × AddOn_full))
AddOn = sum across asset classes of aggregated sensitivity × RW × MF
```

The **Maturity Factor (MF)** is the key MPOR adjustment:
```
MF_unmargined = sqrt(min(M, 1yr) / 1yr)
MF_margined   = 1.5 × sqrt(MPOR / 1yr)
```

The 1.5× factor for margined trades was controversial — it over-penalises frequently-margined netting sets.

### FRTB-CVA (SA-CVA)

SA-CVA prices regulatory capital for CVA risk using a sensitivity-based approach mirroring FRTB Market Risk. Capital = the 99%/10-day ES of CVA, where CVA sensitivity to each risk factor (CS, IR, FX, equity, commodity) is computed and aggregated using supervisory correlations.

Key features:
- **Eligible hedges**: single-name CDS, index CDS, swaptions, FX forwards, equity options — each mapped to specific CVA sensitivities
- **Capital reduction for hedges**: 50–100% capital reduction depending on hedge quality
- **No IMM-CVA for new approvals** under Basel IV (internal models for CVA risk were removed in the 2017 revision and not reinstated in the final Basel IV text)

### The Capital Stack View

From a capital perspective, a bilateral uncollateralised derivative consumes:
1. **CCR capital**: SA-CCR EAD × RW × 8% → capital for default loss
2. **CVA capital**: SA-CVA or BA-CVA → capital for CVA mark-to-market volatility
3. **Market risk capital**: FRTB-SA or IMA → capital for underlying risk factor moves

These three capital charges add up. KVA should reflect all three components, not just CCR capital.

---

## 15. Internal Transfer Pricing

This is where XVA meets the political economy of a bank.

### The XVA Charge Mechanism

When a trading desk executes an OTC derivative, the XVA desk charges them at inception. This charge appears as:
- A day-1 P&L debit on the trading desk
- A day-1 P&L credit on the XVA desk

The trading desk sees a "clean P&L" net of XVA; the XVA desk manages the residual risk.

### What Gets Charged

**CVA**: Always charged. The counterparty's credit spread implies a cost of credit protection that the bank is implicitly providing.

**FVA**: Charged for uncollateralised or partially-collateralised trades. The treasury sets the internal funding rate (IFR) — typically SOFR/ESTR + credit spread + liquidity premium.

**MVA**: Increasingly charged post-2016 for IM-generating trades. The MVA charge is the present value of the funding cost of future IM, which requires the IM simulation described in section 8.

**KVA**: The most controversial. Many desks charge a KVA on new trades, but the definition of "capital cost" is subject to debate — is it cost of equity above risk-free, or above the full bank WACC?

### The Netting Effect and Incremental vs Standalone

**Standalone XVA**: The XVA of the trade if it were in its own netting set. Never realistic for existing counterparties but sometimes used for simplicity.

**Incremental XVA**: The change in total netting-set XVA when the new trade is added. This is the economically correct measure. Incremental CVA can be negative — if the new trade reduces EE (e.g., the client wants to unwind an existing position), the bank's CVA reserve decreases, and the charge should be negative (a rebate).

**Marginal XVA**: XVA allocation to an existing trade using Euler's theorem:
```
XVA_i = (∂XVA_total / ∂λ_i)|_{λ_i=1}
```
where λ_i scales the notional of trade i. Marginal XVA sums to total XVA exactly (Euler homogeneity), so it's used for P&L attribution across existing trades.

### Disputes and Gaming

**Information asymmetry**: Trading desks often know more about specific counterparty risks than the XVA desk. A desk that knows a counterparty is in financial trouble might rush to unwind their CVA-positive trades (locking in the CVA reserve) before the counterparty defaults.

**Threshold arbitrage**: Desks negotiate CSA terms with clients. A smaller threshold reduces CVA (good for XVA desk, reduces charge) but might lose the trade to a competitor. The XVA desk wants tight CSAs; the sales desk wants loose ones to win business.

**KVA debate**: If a trade generates KVA of £100k but the trading desk's P&L on the trade is £200k, they will argue the KVA model is wrong before accepting a 50% reduction in apparent profitability.

### The "Clean" Price vs "All-in" Price

Sales desks quote clients at the "all-in" price = mid-market + bid-ask + XVA. The client sees a single price. The bank internally splits this into:
- Bid-ask spread → trading desk P&L
- XVA charge → XVA desk P&L
- Mid-market → flows to hedge

Competition between dealers is ultimately competition on XVA efficiency. A dealer with better netting, better hedging, or lower capital costs can offer a tighter all-in price.

---

## 16. Technology and Quant Infrastructure

### The Compute Problem

XVA is the most computationally intensive task in derivatives finance. A typical large dealer's XVA run:

- 50,000 simulation paths
- 500 time steps (daily for 2 years, then monthly)
- 5,000 counterparties
- 200,000 trades
- 5–10 stochastic models (rates, FX, equity, credit, commodity)

Naively, this is 50k × 500 × 200k = 5 trillion path-time-trade evaluations. Even at 1μs per evaluation, this is months of compute. In practice, it must complete in hours (overnight batch).

The key optimisations:
1. **Netting set aggregation**: Evaluate per-netting-set (10k–50k), not per-trade
2. **Sparse time grids**: Daily near-term, weekly mid, monthly long-term (170 points, not 2500)
3. **Monte Carlo variance reduction**: Antithetics, quasi-random (Sobol), control variates
4. **Parallelisation**: Multi-threaded simulation, multi-process exposure per netting set
5. **AMC (American Monte Carlo)**: Regress continuation values for Bermudan/callable instruments — avoids nested simulation
6. **GPU acceleration**: Heston, LSM regression — 50–500x speedup for path simulation

### American Monte Carlo (Longstaff-Schwartz in XVA)

Callable/Bermudan instruments require knowing the future exercise value to price at each simulation date — this requires knowing the future price distribution, which is a nested simulation problem. Naive nested Monte Carlo is O(N²) in paths.

AMC solves this via regression:
1. Run the outer simulation paths forward
2. At each exercise date, regress future cashflows on current state variables (rates, spot, vol)
3. Use the regression to approximate future continuation value on any path

In pyxva, `StatefulPricer` is the hook for path-dependent pricing; a full AMC pricer would implement `step()` using pre-fitted regression coefficients from a backward pass.

### The Greeks Challenge

XVA Greeks (for hedging) require bumping market data and re-running the simulation:
- CS01: bump each counterparty's CDS spread by 1bp → re-run CVA → delta
- IR01: bump each rate bucket → re-run EE → delta
- Vega: bump vol surfaces

With 5000 counterparties and 50 rate buckets, full bump-and-revalue requires 5050 full XVA runs. This is infeasible overnight; most firms use:

**Likelihood ratio method (LRM)**: Analytically compute the sensitivity without re-simulation. Exact but requires differentiable models.

**Pathwise differentiation (AAD — Adjoint Algorithmic Differentiation)**: Reverse-mode automatic differentiation through the entire simulation and pricing stack. Computes all Greeks simultaneously in ~3–5× the forward pass cost, regardless of the number of Greeks. This is the state of the art; major investment in AAD toolkits (QuantLib AAD, Murex AAD, NAG).

**Proxy simulation / proxy revaluation**: Pre-compute a polynomial approximation to XVA as a function of risk factors; evaluate the polynomial for bump scenarios.

### Data Requirements

| Input | Source | Update Frequency |
|---|---|---|
| CDS spreads | Bloomberg / MarkIT | Daily |
| Proxy hazard curves | Internal credit model | Weekly |
| IR curves | Bloomberg | Daily (intraday for live) |
| Vol surfaces | Bloomberg / broker | Daily |
| CSA terms | Legal / OTC DR database | Per trade inception |
| Active disputes | Collateral team | Daily |
| IM balances | Custodians | Daily |
| SA-CCR supervisory params | Basel text | Rare changes |
| Internal funding rates (IFR) | Treasury | Daily |

### Model Risk in XVA

The XVA model stack has multiple layers, each with model risk:
1. **Stochastic model**: Heston vs SV, HW1F vs 2F — drives EE shape
2. **Hazard curve**: CDS bootstrap method, proxy mapping — drives CVA
3. **IM model**: SIMM approximation quality — drives MVA
4. **Discount curve**: OIS discounting basis — drives FVA
5. **Correlation**: Cholesky structure across models — drives wrong-way risk

Model risk reserves for XVA are large — typically 10–30% of CVA mark as additional reserve, reflecting uncertainty in EE models and hazard curves.

---

## 17. Open Debates and Unsolved Problems

### 1. FVA and DVA Double-Counting

The Hull-White critique has never been fully resolved. In practice, most banks charge both FVA and DVA, rationalising that DVA is an accounting adjustment (IFRS 13 requirement) while FVA is a real economic cost (treasury funding rate). The theoretical inconsistency remains.

### 2. The KVA Standard

There is no industry consensus on how to define KVA, what discount rate to use, what capital model to assume (SA-CCR? IMM?), or how to handle the stochastic nature of future capital requirements. Firms range from "KVA = CoC × today's EAD × T" to full path-simulated regulatory capital. The divergence in KVA charges between dealers is significant — creating regulatory arbitrage opportunities.

### 3. Close-Out Convention Under ISDA

When a counterparty defaults, the CSA prescribes close-out: the surviving party determines the replacement cost of each trade and nets them. The ISDA 2002 Master Agreement introduced "replacement value" close-out (what a market participant would pay), but the interpretation of "market participant" — does it include DVA? — remains contested in litigation.

The RFC (risk-free close-out) vs CPC (continuation of value) debate in academic XVA literature maps to this legal uncertainty.

### 4. Systemic Wrong-Way Risk

Post-2020, there is growing concern about macro-level wrong-way risk: correlations between sovereign credit spreads and rates (an ECB trade where the ECB's own credit deteriorates as rates fall). The 2016 ISDA SIMM framework tries to capture this via cross-risk-class correlations, but Monte Carlo XVA models typically assume low correlation between interest rates and credit spreads — an assumption that may break down in stress.

### 5. Clearing vs Bilateral: The True XVA Comparison

The "cleared XVA is lower" narrative is often true for vanilla products but wrong for structured ones. CCP IM methodology (SPAN or HVaR for futures; PRISMA or SIMM for OTC) may demand higher IM than bilateral SIMM for bespoke payoffs. The CCP default fund contribution is also a KVA cost that is often ignored in naive comparisons.

### 6. XVA Under Stress

Standard XVA models use market-implied, risk-neutral hazard rates calibrated to current CDS spreads. In a systemic credit event (2008-style), CDS spreads blow out, CVA P&L losses are massive, and hedge ratios (CS01) were underestimated because the model assumed smaller spread moves. Stress-testing XVA — using historical scenarios (2008, 2011 Eurozone, 2020 COVID) — is still not standardised. Firms have internal stress CVA reserves but the calculation methodology varies widely.

---

## Further Reading

**Books**:
- Gregory, Jon — *Counterparty Credit Risk and Credit Value Adjustment* (2nd ed., 2012) — the canonical reference
- Green, Andrew — *XVA: Credit, Funding and Capital Valuation Adjustments* (2015) — comprehensive unified framework
- Lichters, Roland / Stamm, Roland / Gallagher, Donal — *Modern Derivatives Pricing and Credit Exposure Analysis* (2016) — practitioner focus with code

**Papers**:
- Hull, John & White, Alan — "The FVA Debate" (Risk, 2012) — the original FVA critique
- Burgard & Kjaer — "In the Balance" (Risk, 2011) — rigorous PDE derivation of CVA+DVA+FVA
- Albanese, Andersen, Iabichino — "FVA: Accounting and Risk Management" (Risk, 2015) — unified economic framework
- Green & Kenyon — "MVA: Initial Margin Valuation Adjustment" (Risk, 2015) — first systematic MVA treatment

**Regulatory**:
- BCBS CRE52: SA-CCR for counterparty credit risk
- BCBS MAR50: FRTB-CVA
- BCBS-IOSCO: Final Framework for Margin Requirements for Non-Centrally Cleared Derivatives (2015)
- ISDA SIMM Methodology (current version published annually)
