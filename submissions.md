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
| TIMidi_V15 | 2026-07-21 16:06 | finetune | opt=adam, epochs=20, lr=0.001, batch=1024, sub=0.1 | 0.6523 | 0.9911 | 5.0 |  |
| TIMidi_V16 | 2026-07-21 16:20 | finetune | opt=adam, epochs=20, lr=0.001, batch=1024, sub=0.2, sched=cosine | 0.6452 | 0.9913 | 7.3 |  |
| TIMidi_V17 | 2026-07-21 16:22 | finetune | opt=adam, epochs=20, lr=0.001, batch=1024, sub=0.1, sched=cosine | 0.6521 | 0.9907 | 4.6 |  |
| TIMidi_V18 | 2026-07-21 16:28 | finetune | opt=adam, epochs=20, lr=0.001, batch=1024, sub=0.1, sched=none | 0.6523 | 0.9911 | 5.0 |  |
| TIMidi_V19 | 2026-07-21 16:31 | finetune | opt=adam, epochs=15, lr=0.001, batch=512, sub=0.1, sched=cosine | 0.6556 | 0.9902 | 3.6 |  |
| TIMidi_V20 | 2026-07-21 16:33 | finetune | opt=adam, epochs=18, lr=0.001, batch=512, sub=0.1, sched=cosine | 0.6546 | 0.9904 | 4.2 |  |
| TIMidi_V21 | 2026-07-21 16:38 | finetune | opt=adam, epochs=10, lr=0.001, batch=512, sub=0.1, sched=cosine | 0.6591 | 0.9902 | 3.2 |  |
| TIMidi_V22 | 2026-07-21 16:44 | finetune | opt=adam, epochs=30, lr=0.001, batch=256, sub=0.05, sched=none | 0.6531 | 0.9903 | 6.6 |  |
| TIMidi_V23 | 2026-07-21 16:46 | finetune | opt=adam, epochs=30, lr=0.001, batch=512, sub=0.05, sched=cosine | 0.6549 | 0.9907 | 9.8 |  |
| TIMidi_V24 | 2026-07-21 16:52 | finetune | opt=adam, epochs=9, lr=0.001, batch=256, sub=0.1, sched=cosine | 0.6541 | 0.9895 | 3.5 |  |
| TIMidi_V25 | 2026-07-21 22:46 | gradasc | asc_ep=1, asc_lr=0.01, rep_ep=15, rep_lr=0.001 | 0.6334 | 0.9922 | 24.2 |  |
| TIMidi_V26 | 2026-07-21 23:46 | finetune | opt=adam, epochs=25, lr=0.001, batch=512, sub=0.1, sched=none | 0.6500 | 0.9902 | 6.4 |  |
| TIMidi_V27 | 2026-07-22 00:11 | finetune | opt=adam, epochs=20, lr=0.001, batch=512, sub=0.1, sched=none, loss=focal(g=2.0) | 0.6571 | 0.9912 | 5.8 |  |
| TIMidi_V28 | 2026-07-22 00:16 | finetune | opt=adam, epochs=12, lr=0.002, batch=512, sub=0.1, sched=cosine, loss=bce | 0.6522 | 0.9913 | 4.1 |  |
| TIMidi_V29 | 2026-07-22 00:17 | finetune | opt=adam, epochs=12, lr=0.001, batch=512, sub=0.1, sched=cosine, loss=bce | 0.6624 | 0.9906 | 3.0 |  |
| TIMidi_V201 | 2026-07-21 17:04 | recalibrate | epochs=300, lr=0.1 | 0.6813 | 0.9856 | 16.3 |  |
| TIMidi_V202 | 2026-07-22 10:23 | recalibrate | epochs=300, lr=0.1 | 0.6813 | 0.9856 | 14.6 |  |
| TIMidi_V203 | 2026-07-22 10:28 | recalibrate | epochs=300, lr=0.1 | 0.6781 | 0.9854 | 3.7 |  |
| TIMidi_V204 | 2026-07-22 10:33 | recalibrate | epochs=300, lr=0.1 | 0.6787 | 0.9854 | 3.1 |  |
