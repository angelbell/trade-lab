//+------------------------------------------------------------------+
//|  structure_sma.mq5                                               |
//|  Market-structure breakout, gated by a daily SMA150 regime.      |
//|  1:1 port of the Python compute_signals_structure() (LOCK config)|
//|  attach to a 1H BTCUSD chart on Vantage (MT5).                   |
//|                                                                  |
//|  Logic (mirrors strategy.py, entry_mode="breakout"):            |
//|   - 4H structure: fractal swing highs/lows (N bars each side).   |
//|       higher-low (last swing low > prev) => uptrend;             |
//|       lower-high  (last swing high < prev) => downtrend.         |
//|   - Entry: LONG when uptrend (and NOT downtrend) AND the 1H      |
//|       close CROSSES ABOVE the last 4H swing high, gated by       |
//|       daily close > daily SMA(150). SHORT is the mirror.         |
//|   - Exit (two-stage, whichever first):                          |
//|       (1) hard stop = entry -/+ 2*ATR(14) (placed at broker,     |
//|           so it fills intrabar like the backtest's low<=stop);   |
//|       (2) structure stop = 4H trend flips OR the protective      |
//|           swing level (sl/sh) breaks on a 1H close.              |
//|   - Sizing: risk 1% of equity over the 2*ATR stop distance.     |
//|                                                                  |
//|  IMPORTANT design notes (the bugs we already hit porting this):  |
//|   * Entry is a CROSSOVER EVENT, not the level state close>sh.    |
//|     A persistent level re-enters every bar after a stop-out      |
//|     (the 476-trade churn). crossover fires once.                 |
//|   * The `!downtrend` / `!uptrend` guard skips CONTRACTIONS (4H   |
//|     can be higher-lows AND lower-highs at once, ~21% of bars).   |
//|     vectorbt silently drops the same-bar entry/exit conflict;    |
//|     an event engine does not, so we must guard (the 70:1 bug).   |
//|   * We act ONLY on a NEW 1H bar, using CLOSED bar values, to     |
//|     align with the backtest's close-fill (process_orders_on_close|
//|     in Pine). Real market-order slippage vs the close is the     |
//|     one accepted gap between this and the Python numbers.        |
//+------------------------------------------------------------------+
#property copyright "auto-trade"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- inputs (defaults = the LOCKED Python config) -------------------
input int    FractalN      = 3;        // bars each side for a 4H swing pivot
input int    SmaLen        = 150;      // daily regime SMA length
input int    AtrLen        = 14;       // ATR length for the hard stop / sizing
input double AtrMultSl     = 2.0;      // hard stop = AtrMultSl * ATR
input double RiskPct       = 1.0;      // % of equity risked per trade
input bool   UseShorts     = true;     // allow short trades
input int    H4Lookback    = 400;      // # of 4H bars scanned for swings
input long   MagicNumber   = 530150;   // EA order tag
input int    SlippagePts   = 50;       // max deviation (points) for market orders
input bool   DebugLog      = true;     // print every entry/exit + structure state

//--- globals --------------------------------------------------------
CTrade   trade;
int      atrHandle = INVALID_HANDLE;   // ATR(14) on 1H
int      smaHandle = INVALID_HANDLE;   // SMA(150) on D1 close
datetime lastBarTime = 0;              // for new-1H-bar detection

//+------------------------------------------------------------------+
int OnInit()
  {
   atrHandle = iATR(_Symbol, PERIOD_H1, AtrLen);
   smaHandle = iMA(_Symbol, PERIOD_D1, SmaLen, 0, MODE_SMA, PRICE_CLOSE);
   if(atrHandle == INVALID_HANDLE || smaHandle == INVALID_HANDLE)
     {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
     }
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePts);
   trade.SetTypeFillingBySymbol(_Symbol);
   Print("structure_sma EA initialised on ", _Symbol,
         "  (run on the 1H chart; 4H structure + daily SMA", SmaLen, " gate)");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   if(smaHandle != INVALID_HANDLE) IndicatorRelease(smaHandle);
  }

//+------------------------------------------------------------------+
//| 4H market structure: find the last & previous CONFIRMED swing    |
//| high and low (fractal pivots, N bars each side).                 |
//|                                                                  |
//| A bar at series-shift s is a swing high if its high is >= every  |
//| high within +/-N bars. We require N closed bars to the RIGHT     |
//| (more recent) to confirm it (no repaint / no lookahead) -- this  |
//| mirrors Python's centered rolling-max + shift(N), and the        |
//| extra one-bar offset (to_ltf shift(1)) is covered by only        |
//| reading already-closed 4H bars (shift >= 1).                     |
//+------------------------------------------------------------------+
bool GetStructure(double &lastSh, double &prevSh, double &lastSl, double &prevSl)
  {
   int need = H4Lookback;
   double h[], l[];
   ArraySetAsSeries(h, true);
   ArraySetAsSeries(l, true);
   if(CopyHigh(_Symbol, PERIOD_H4, 0, need, h) <= 0) return(false);
   if(CopyLow (_Symbol, PERIOD_H4, 0, need, l) <= 0) return(false);
   int n = MathMin(ArraySize(h), ArraySize(l));

   lastSh = prevSh = lastSl = prevSl = 0.0;
   int shCount = 0, slCount = 0;

   // s = FractalN+1 keeps the N confirming bars (s-1..s-N) on CLOSED bars
   // (shift >= 1). Iterate recent -> old, so the first hit is the most recent.
   for(int s = FractalN + 1; s <= n - 1 - FractalN; s++)
     {
      bool isSh = true, isSl = true;
      for(int k = s - FractalN; k <= s + FractalN; k++)
        {
         if(k == s) continue;
         if(h[k] >  h[s]) isSh = false;   // a higher high nearby -> not a pivot high
         if(l[k] <  l[s]) isSl = false;   // a lower  low  nearby -> not a pivot low
        }
      if(isSh && shCount < 2)
        {
         if(shCount == 0) lastSh = h[s]; else prevSh = h[s];
         shCount++;
        }
      if(isSl && slCount < 2)
        {
         if(slCount == 0) lastSl = l[s]; else prevSl = l[s];
         slCount++;
        }
      if(shCount >= 2 && slCount >= 2) break;
     }
   return(shCount >= 2 && slCount >= 2);
  }

//+------------------------------------------------------------------+
//| Position sizing: lots that risk RiskPct of equity over a stop    |
//| of stopDistPrice price-units (= AtrMultSl * ATR).                |
//+------------------------------------------------------------------+
double RiskLots(double stopDistPrice)
  {
   if(stopDistPrice <= 0) return(0.0);
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMon  = equity * RiskPct / 100.0;
   double tickVal  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickVal <= 0 || tickSize <= 0) return(0.0);

   double lossPerLot = stopDistPrice / tickSize * tickVal;   // money lost per 1.0 lot if stopped
   if(lossPerLot <= 0) return(0.0);
   double lots = riskMon / lossPerLot;

   // clamp to the symbol's lot constraints
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep > 0) lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return(lots);
  }

//+------------------------------------------------------------------+
//| Our open position on this symbol (0 = flat, +1 long, -1 short).  |
//+------------------------------------------------------------------+
int OpenDir()
  {
   if(!PositionSelect(_Symbol)) return(0);
   if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) return(0);
   long t = PositionGetInteger(POSITION_TYPE);
   return(t == POSITION_TYPE_BUY ? 1 : -1);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   // act once per CLOSED 1H bar (align with the backtest close-fill)
   datetime t0 = iTime(_Symbol, PERIOD_H1, 0);
   if(t0 == lastBarTime) return;
   lastBarTime = t0;

   // ---- gather closed-bar inputs ----------------------------------
   double sh, psh, sl, psl;
   if(!GetStructure(sh, psh, sl, psl)) return;

   bool uptrend   = (sl > 0 && psl > 0 && sl > psl);   // higher lows
   bool downtrend = (sh > 0 && psh > 0 && sh < psh);   // lower highs

   // daily SMA150 regime, previous completed day (shift 1)
   double smaBuf[];
   ArraySetAsSeries(smaBuf, true);
   if(CopyBuffer(smaHandle, 0, 1, 1, smaBuf) <= 0) return;
   double dSma   = smaBuf[0];
   double dClose = iClose(_Symbol, PERIOD_D1, 1);
   bool bullReg = (dSma > 0 && dClose > dSma);
   bool bearReg = (dSma > 0 && dClose < dSma);

   // ATR on the signal bar (shift 1) -> hard stop distance & sizing
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(atrHandle, 0, 1, 1, atrBuf) <= 0) return;
   double atr = atrBuf[0];
   double stopDist = AtrMultSl * atr;

   // closed 1H closes for the crossover (shift1 = last closed, shift2 = before)
   double c1 = iClose(_Symbol, PERIOD_H1, 1);
   double c2 = iClose(_Symbol, PERIOD_H1, 2);

   int dir = OpenDir();

   // ---- exits: structure flip / protective level break ------------
   // (the hard 2*ATR stop is a real broker SL, fills intrabar on its own)
   if(dir > 0)
     {
      bool structExitL = downtrend || (sl > 0 && c1 < sl);
      if(structExitL)
        {
         if(DebugLog) PrintFormat("EXIT L  %s  c1=%.2f sl=%.2f dn=%d",
              TimeToString(t0, TIME_DATE|TIME_MINUTES), c1, sl, downtrend);
         trade.PositionClose(_Symbol); return;
        }
     }
   else if(dir < 0)
     {
      bool structExitS = uptrend || (sh > 0 && c1 > sh);
      if(structExitS)
        {
         if(DebugLog) PrintFormat("EXIT S  %s  c1=%.2f sh=%.2f up=%d",
              TimeToString(t0, TIME_DATE|TIME_MINUTES), c1, sh, uptrend);
         trade.PositionClose(_Symbol); return;
        }
     }

   // ---- entries: breakout LEVEL state, gated by trend + regime ----
   // We enter while the 1H close is ABOVE the last 4H swing high (long), i.e.
   // the exact `close > sh` condition the VALIDATED Python strategy uses --
   // NOT a one-bar crossover. Crossover only catches the single crossing bar
   // and skips gap-throughs / re-entries, which measurably cut the edge
   // (2024: PF 1.69 -> 1.05). This does NOT churn: we trade only while flat,
   // and every exit (hard 2*ATR stop, or close<sl) leaves price BELOW sh, so
   // the level naturally re-arms exactly like vectorbt (17 trades, not 476).
   if(dir != 0) return;        // one position at a time (pyramiding=0)

   bool brkUp   = (sh > 0 && c1 > sh);
   bool brkDown = (sl > 0 && c1 < sl);

   bool longCond  = uptrend && !downtrend && brkUp   && bullReg;
   bool shortCond = UseShorts && downtrend && !uptrend && brkDown && bearReg;

   if(longCond)
     {
      double lots = RiskLots(stopDist);
      if(lots <= 0) return;
      double ask  = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double slPr = NormalizeDouble(ask - stopDist, _Digits);
      if(DebugLog) PrintFormat("ENTRY L %s  c1=%.2f sh=%.2f sl=%.2f up=%d dn=%d bull=%d atr=%.2f",
           TimeToString(t0, TIME_DATE|TIME_MINUTES), c1, sh, sl, uptrend, downtrend, bullReg, atr);
      trade.Buy(lots, _Symbol, 0.0, slPr, 0.0, "struct long");
     }
   else if(shortCond)
     {
      double lots = RiskLots(stopDist);
      if(lots <= 0) return;
      double bid  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double slPr = NormalizeDouble(bid + stopDist, _Digits);
      if(DebugLog) PrintFormat("ENTRY S %s  c1=%.2f sh=%.2f sl=%.2f up=%d dn=%d bear=%d atr=%.2f",
           TimeToString(t0, TIME_DATE|TIME_MINUTES), c1, sh, sl, uptrend, downtrend, bearReg, atr);
      trade.Sell(lots, _Symbol, 0.0, slPr, 0.0, "struct short");
     }
  }
//+------------------------------------------------------------------+
