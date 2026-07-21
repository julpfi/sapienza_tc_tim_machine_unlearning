# Submissions log

| Version | Date | Method | Params | P@10 | MIA score | Time(s) | Notes |
|---|---|---|---|---|---|---|---|
| TIMidi_V1 | 2026-07-20 19:29 | finetune | epochs=5, lr=1e-2, split=70/15/15 | 0.6045 | 0.988 | 9 | baseline; leaderboard 82.640 (prec .633 / mia .997 / time .925) |
| TIMidi_V2 | 2026-07-20 19:xx | ssd | alpha=5, lam=2, split=70/15/15 | 0.6401 | 0.988 | 3 | SSD tuned, eval on test split |
| TIMidi_V3 | 2026-07-20 19:xx | gradasc | ascent1/repair2 (default), split=70/15/15 | 0.5924 | 0.986 | 4 | gradient ascent, weaker than baseline |
| TIMidi_V4 | 2026-07-20 19:xx | ssd | alpha=5, lam=2, split=85/15 | 0.6358 | 0.992 | 2 | SSD tuned, more train data (85%) |
| TIMidi_V5 | 2026-07-21 15:00 | fisher | sigma=1e-06, eps=0.0001 | 0.6374 | 0.9923 | 2.0 |  |
| TIMidi_V6 | 2026-07-21 15:25 | finetune | opt=adam, epochs=20, lr=0.001 | 0.6426 | 0.9919 | 41.2 |  |
| TIMidi_V7 | 2026-07-21 15:27 | finetune | opt=adam, epochs=40, lr=0.001 | 0.6486 | 0.9916 | 73.1 |  |
| TIMidi_V8 | 2026-07-21 15:29 | finetune | opt=adam, epochs=55, lr=0.001 | 0.6427 | 0.9915 | 105.4 |  |
| TIMidi_V9 | 2026-07-21 15:41 | finetune | opt=adam, epochs=20, lr=0.001 | 0.6439 | 0.9912 | 7.6 |  |
| TIMidi_V10 | 2026-07-21 15:45 | finetune | opt=adam, epochs=15, lr=0.001 | 0.6468 | 0.9911 | 5.4 |  |
| TIMidi_V11 | 2026-07-21 15:50 | finetune | opt=adam, epochs=20, lr=0.001 | 0.6523 | 0.9911 | 4.7 |  |
| TIMidi_V12 | 2026-07-21 15:54 | finetune | opt=adam, epochs=20, lr=0.01 | 0.6081 | 0.9907 | 5.5 |  |
| TIMidi_V13 | 2026-07-21 15:56 | finetune | opt=adam, epochs=20, lr=0.01 | 0.6081 | 0.9907 | 5.9 |  |
| TIMidi_V14 | 2026-07-21 15:59 | finetune | opt=adam, epochs=15, lr=0.001 | 0.6468 | 0.9911 | 5.3 |  |
