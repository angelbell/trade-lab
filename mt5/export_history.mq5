//+------------------------------------------------------------------+
//|  export_history.mq5  --  dump OHLCV to CSV for offline validation |
//|                                                                   |
//|  A SCRIPT (not an EA). Run it on a BTCUSD chart in MT5 to export   |
//|  the broker's own H1 history to a CSV, so we can re-run the WFO /  |
//|  holdout pipeline on the data we ACTUALLY trade (Vantage), instead |
//|  of Binance. Times are in BROKER SERVER time -- that is exactly    |
//|  what the live EA sees, so the 4H / daily bar boundaries match.    |
//|                                                                    |
//|  Usage: drag onto a BTCUSD,H1 chart. The file lands in the data    |
//|  folder under  MQL5\Files\<InpFile>.                               |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input ENUM_TIMEFRAMES InpTF   = PERIOD_H1;             // timeframe to export
input int             InpBars = 200000;                // bars back to export (take everything)
input string          InpFile = "";                    // blank = auto-name vantage_<symbol>_<tf>.csv

string tf_str(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:  return "m1";
      case PERIOD_M5:  return "m5";
      case PERIOD_M15: return "m15";
      case PERIOD_M30: return "m30";
      case PERIOD_H1:  return "h1";
      case PERIOD_H4:  return "h4";
      case PERIOD_D1:  return "d1";
      case PERIOD_W1:  return "w1";
      default:         return "tf" + IntegerToString(PeriodSeconds(tf) / 60);
     }
  }

void OnStart()
  {
   string fname = InpFile;
   if(StringLen(fname) == 0)
     {
      string sym = _Symbol;
      StringToLower(sym);
      fname = "vantage_" + sym + "_" + tf_str(InpTF) + ".csv";
     }

   MqlRates r[];
   ArraySetAsSeries(r, true);
   int got = CopyRates(_Symbol, InpTF, 0, InpBars, r);
   if(got <= 0)
     {
      Print("CopyRates failed (err=", GetLastError(),
            "). Scroll the chart left to load history, then retry.");
      return;
     }

   int fh = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("FileOpen failed (err=", GetLastError(), ")");
      return;
     }

   FileWrite(fh, "time", "open", "high", "low", "close", "tick_volume");
   // oldest -> newest (r is series, so iterate from the back)
   for(int i = got - 1; i >= 0; i--)
      FileWrite(fh,
                TimeToString(r[i].time, TIME_DATE | TIME_MINUTES),
                DoubleToString(r[i].open,  _Digits),
                DoubleToString(r[i].high,  _Digits),
                DoubleToString(r[i].low,   _Digits),
                DoubleToString(r[i].close, _Digits),
                (long)r[i].tick_volume);
   FileClose(fh);

   PrintFormat("Exported %d bars (%s -> %s) to MQL5\\Files\\%s",
               got,
               TimeToString(r[got - 1].time, TIME_DATE),
               TimeToString(r[0].time, TIME_DATE),
               fname);
  }
//+------------------------------------------------------------------+
