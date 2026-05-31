# Binance Strategy Simulator (محاكي استراتيجيات بينانس)

إطار عمل بايثون لمحاكاة استراتيجيات التداول على سوق **Binance Spot**، يدعم:

- **Backtesting** — اختبار الاستراتيجية على بيانات الشموع التاريخية لقياس الأداء.
- **Paper Trading** — محاكاة لحظية بأموال وهمية على بيانات السوق الحية (عبر سحب الشموع من واجهة بينانس العامة).
- **إطار استراتيجيات مرن** — أضف أي استراتيجية بوراثة كلاس واحد وإرجاع إشارة `BUY/SELL/HOLD`.

النواة تعتمد فقط على مكتبة `requests` (المؤشرات الفنية مكتوبة بـ Python نقي)، فلا حاجة لمكتبات ثقيلة.

---

## ⚠️ تنبيه مهم

هذا المشروع **للمحاكاة والتعلم فقط**. لا يقوم بأي تداول حقيقي ولا يرسل أوامر بأموال حقيقية.
- جلب البيانات يستخدم واجهة بينانس **العامة** (لا يحتاج مفاتيح API).
- التداول الوهمي (Paper) يحاكي المحفظة محلياً على بيانات الأسعار الحية.
- التداول بالعملات الرقمية ينطوي على مخاطر عالية. لا شيء هنا نصيحة مالية.

---

## التثبيت

```bash
pip install -r requirements.txt
```

المتطلب الوحيد للنواة هو `requests`. (`pandas`/`numpy` اختيارية للتحليل فقط).

---

## الاستخدام السريع

### 1) اختبار تاريخي (Backtest)

```bash
python -m binance_sim backtest \
  --symbol BTCUSDT --interval 1h --limit 1000 \
  --strategy ma_crossover --params '{"fast":20,"slow":50}' \
  --cash 10000 --fee 0.001
```

### 2) محاكاة لحظية (Paper Trading)

```bash
python -m binance_sim paper \
  --symbol BTCUSDT --interval 1m \
  --strategy rsi --params '{"period":14,"low":30,"high":70}' \
  --cash 10000 --fee 0.001 --state state.json
```

يحفظ الحالة في `state.json` ليُستأنف بعد التوقف (مفيد في البيئات المؤقتة).

### 3) قائمة الاستراتيجيات المتاحة

```bash
python -m binance_sim list
```

---

## كتابة استراتيجية جديدة

أنشئ ملفاً في `binance_sim/strategies/` يرث من `Strategy`:

```python
from binance_sim.strategy import Strategy, Signal
from binance_sim.indicators import ema

class MyStrategy(Strategy):
    name = "my_strategy"

    def __init__(self, period: int = 21):
        self.period = period

    @property
    def warmup(self) -> int:
        # عدد الشموع المطلوبة قبل أن تنتج الاستراتيجية إشارة موثوقة
        return self.period + 1

    def generate_signal(self, candles) -> Signal:
        closes = [c.close for c in candles]
        line = ema(closes, self.period)
        if closes[-1] > line[-1]:
            return Signal.BUY
        if closes[-1] < line[-1]:
            return Signal.SELL
        return Signal.HOLD
```

ثم سجّلها في `binance_sim/strategies/__init__.py`.

كل استراتيجية تستقبل **كامل تاريخ الشموع حتى الشمعة المغلقة الحالية** وتُرجع إشارة واحدة.
المحرك يتكفّل بإدارة المحفظة، الرسوم، وحجم الصفقة.

---

## بنية المشروع

```
binance_sim/
├── client.py          عميل بيانات بينانس العامة (REST + ترقيم تلقائي)
├── indicators.py      مؤشرات فنية بـ Python نقي (SMA, EMA, RSI, ...)
├── strategy.py        كلاس Strategy الأساسي + enum الإشارات
├── portfolio.py       محاكاة المحفظة (نقد/مركز/رسوم/صفقات)
├── backtest.py        محرك الاختبار التاريخي + مقاييس الأداء
├── paper.py           محرك المحاكاة اللحظية مع حفظ الحالة
├── strategies/        الاستراتيجيات الجاهزة
│   ├── ma_crossover.py
│   └── rsi.py
└── __main__.py        واجهة سطر الأوامر (CLI)
```

## المقاييس المحسوبة في الاختبار التاريخي

- العائد الكلي ومقارنته بـ Buy & Hold
- عدد الصفقات ونسبة الرابحة منها (Win rate)
- أقصى تراجع (Max Drawdown)
- نسبة شارب التقريبية (Sharpe)
