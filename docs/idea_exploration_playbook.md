# Idea-exploration playbook — raw hunch → honest verdict, fast

How to take "I'm thinking of trading X" to a defensible LEAD / KILL without
wasting effort or fooling ourselves. This is the **front end** to the
`CLAUDE.md` falsification checklist — it does not repeat it, it feeds it.

> One-liner: **screen cheap, kill fast, build only survivors, then audit, then
> forward-test.** Validate-don't-falsify is the failure mode; this avoids it.

## The loop (per idea)

1. **Demand a mechanism first.** *Why* would this edge exist (liquidity, vol
   clustering, positioning, gap dynamics)? No plausible mechanism → don't test.
   And distrust the stated mechanism: check it. (e.g. "Monday = high liquidity"
   is *false* for gold — Monday is thin; the real peak is Tue–Thu London/NY.)

2. **Cheapest KILL screen BEFORE writing a strategy.** Reduce the idea to a
   data question answerable in one script, no entries/exits yet:
   - split forward returns by the condition vs a baseline of all bars,
   - look at mean/median/win% **and** per-year stability,
   - price in luck with a **permutation / bootstrap null**.
   If the condition isn't distinguishable from random here, stop — no strategy
   built on it can rescue it. (Kills most ideas in minutes.)

3. **Only survivors get a real backtest.** Mechanize precisely, then run the
   full `CLAUDE.md` **falsification checklist** (all-signals base → win/RR-breakeven
   → IS/OOS → ±1 parameter sweep = plateau-not-spike → per-year spread → cost →
   selection-trap → beta/short-side → no-lookahead).

4. **Decompose along the three axes that usually decide it:**
   - **Timeframe** — prior: **4H is the structural sweet spot**; 1H shreds the
     edge to noise, 1D goes thin. Test 1H/4H/1D; don't assume daily.
   - **Direction** — does it survive the short side, or is it just long-only beta?
   - **Exit** — fixed-RR vs trailing. (BTC: distant fixed targets harvest the
     expansion; trailing gets shaken out on retraces. Higher RR ⇒ higher totR
     but longer losing streaks — check DD/consec-loss, not just totR.)

5. **MEASURE overfit** with `research/overfit_audit.py` methodology — Deflated
   Sharpe (survives the trial-count haircut?), PBO/CSCV (does parameter pick
   generalize, or use plateau defaults?), bootstrap null (edge ≠ luck?). A pass
   = quantified confidence, NOT proof. It can't see future regimes.

6. **Verdict honestly.** Front-loaded / era-dead / one-side beta / param-fragile
   ⇒ KILL. Survives audit but marginal ⇒ **paper / forward-test candidate, NOT a
   confirmed book member** — live-forward (demo) is the arbiter.

7. **Log every trial** (incl. kills) in `docs/scalp_research_log.md`. The more
   we try, the higher the significance bar (multiple comparisons). Hidden trials
   under-count N and hide overfit.

8. **Save a reproducible script** for any survivor so the numbers regenerate
   from disk, not from a chat transcript.

## How to drive it (collaboration pattern that worked)

The human supplies, each turn: (a) the **mechanism hypothesis**, (b) a **next
axis to probe** — the single most valuable steer is often *"try a smaller/other
timeframe."* The agent supplies: cheap screens, the checklist, honest kills, and
the audit. Don't defend losers; kill them and move on.

### Reusable opening prompt

> Falsify this idea, don't validate it. ① cheap KILL screen (no strategy code)
> before building — split returns by the condition vs baseline, with a
> permutation/bootstrap null. ② if it survives, full backtest with IS/OOS +
> per-year + ±1 parameter sweep + cost. ③ decompose timeframe / direction /
> exit. ④ run it through `overfit_audit` (DSR / PBO / bootstrap). ⑤ save a
> reproducible script and confirm the numbers regenerate. Judge it as a
> forward-test candidate, not a confirmed edge — be blunt about kills.

## Worked example (this is what "good" looks like)

Session 2026-06: hunch "gold Monday-open entry (Monday = more liquidity)".
- KILL screen exposed the premise as backwards (Monday is thin) and the blind
  long as 2007–2011 beta; gap-**fade** was real but **era-dead** (2016–26 ≈ 0);
  continuation = generic trend beta; BTC→gold lead-lag corr < 0.05 → KILL.
- The one survivor came from a *seed idea* (vol-**squeeze** breakout), promoted
  on the human's *"look at smaller timeframes too"* steer: **BTC 4H squeeze
  breakout** — parameter-robust, both-direction, OOS-stable; audit marginal
  (DSR borderline past ~25 trials, PBO ≈ 0.5) ⇒ **forward-test candidate**.
- Reproducible script (the reference implementation for this playbook):
  **`../mt5-mcp/research/btc_4h_squeeze.py`** — regenerates the full backtest /
  timeframe / direction / exit / stress / audit tables from the Vantage CSV.
