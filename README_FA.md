# طبقه‌بندی رادیوگرافی دندانی (DenseNet)

طبقه‌بندی بافت دندان در تصاویر رادیوگرافی به سه کلاس **عاج (dentin)**، **مینا (enamel)** و **پالپ (pulp)** با DenseNet121.

## چه کاری انجام می‌دهد

- آموزش یک طبقه‌بند سه‌کلاسه DenseNet121 روی پچ‌های بخش‌بندی‌شده دندان
- تشخیص با پنجره لغزان روی رادیوگراف کامل
- پیش‌بینی دسته‌ای پوشه‌های تست (`dentin` / `enamel` / `pulp`)
- محاسبه Precision، Recall، Accuracy و F1 از فایل‌های CSV پیش‌بینی

## پیش‌نیازها

- Python 3.10+
- PyTorch با CUDA، Apple MPS یا CPU
- وابستگی‌ها در `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ساختار پروژه

```
densNet/
├── train.py                      # آموزش DenseNet121
├── detect_simple.py              # تشخیص پنجره لغزان (تعاملی)
├── predict_dentin_test.py        # پیش‌بینی dentin_test → CSV
├── predict_enamel_test.py        # پیش‌بینی enamel_test → CSV
├── predict_pulp_test.py          # پیش‌بینی pulp_test → CSV
├── evaluate_test_predictions.py  # معیارها از CSVهای پیش‌بینی
├── requirements.txt
├── slm/                          # چک‌پوینت‌های ذخیره‌شده مدل
├── image-testing/
│   ├── dentin_test/
│   ├── enamel_test/
│   └── pulp_test/
├── segmented_dental_adiography/  # train / valid / test برای آموزش
└── test_predictions/             # خروجی CSV + گزارش ارزیابی
```

## چک‌پوینت مدل

اسکریپت‌های استنتاج از این فایل استفاده می‌کنند:

```text
slm/resolution_best_densenet_model.pth
```

کلاس‌ها (به ترتیب ایندکس): `0 = dentin`، `1 = enamel`، `2 = pulp`.

## آموزش

ساختار دیتاست:

```text
segmented_dental_adiography/
├── train/{dentin,enamel,pulp}/
├── valid/{dentin,enamel,pulp}/
└── test/{dentin,enamel,pulp}/
```

```bash
python train.py
```

بهترین چک‌پوینت را ذخیره می‌کند و نمودارهای آموزش و ماتریس درهم‌ریختگی را می‌نویسد.

## تشخیص (تصویر کامل)

اسکن تعاملی با پنجره لغزان و کادرهای رنگی:

- قرمز = عاج (dentin) · سبز = مینا (enamel) · آبی = پالپ (pulp)

```bash
python detect_simple.py
# مسیر تصویر را وقتی پرسیده شد وارد کنید
```

## تصاویر تست خارجی (`image-testing/`)

پچ‌های بافتی جدا برای ارزیابی دسته‌ای (متفاوت از `segmented_dental_adiography/` که برای آموزش استفاده می‌شود).

```text
image-testing/
├── dentin_test/   # ۷۵ تصویر — برچسب واقعی: dentin (عاج)
├── enamel_test/   # ۷۵ تصویر — برچسب واقعی: enamel (مینا)
└── pulp_test/     # ۷۵ تصویر — برچسب واقعی: pulp (پالپ)
```

| پوشه | کلاس | تعداد |
|--------|-------|-------|
| `dentin_test/` | dentin (عاج) | ۷۵ |
| `enamel_test/` | enamel (مینا) | ۷۵ |
| `pulp_test/` | pulp (پالپ) | ۷۵ |
| **جمع** | | **۲۲۵** |

فرمت‌های پشتیبانی‌شده: `.png`، `.jpg`، `.jpeg` (با هر حروف بزرگ/کوچک).

## پیش‌بینی دسته‌ای تست

هر اسکریپت همه تصاویر پوشه متناظر در `image-testing/*_test` را طبقه‌بندی می‌کند و یک CSV در `test_predictions/` می‌نویسد.

ستون‌های CSV (هم‌شکل با خروجی‌های ارزیابی خارجی):

`file,class_name,target,target_label,pred_idx,pred_label,prob_positive`

`prob_positive` احتمال کلاس هدف همان اسکریپت است (P(dentin)، P(enamel) یا P(pulp)).

```bash
python predict_dentin_test.py
python predict_enamel_test.py
python predict_pulp_test.py
```

خروجی‌ها:

| اسکریپت | ورودی | خروجی |
|--------|-------|--------|
| `predict_dentin_test.py` | `image-testing/dentin_test/` | `test_predictions/dentin_test_predictions.csv` |
| `predict_enamel_test.py` | `image-testing/enamel_test/` | `test_predictions/enamel_test_predictions.csv` |
| `predict_pulp_test.py` | `image-testing/pulp_test/` | `test_predictions/pulp_test_predictions.csv` |

## معیارهای ارزیابی

پس از وجود سه CSV پیش‌بینی:

```bash
python evaluate_test_predictions.py
```

از این فرمول‌ها استفاده می‌کند:

| معیار | فرمول |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |

خروجی‌ها:

- `test_predictions/evaluation_metrics.csv`
- `test_predictions/evaluation_report.txt`
- همچنین `test_predictions/evaluation_report.md`

### آخرین نتایج تست خارجی (۲۲۵ تصویر)

| کلاس | Precision | Recall | Accuracy | F1 |
|-------|-----------|--------|----------|-----|
| dentin (عاج) | 95.38% | 82.67% | 92.89% | 88.57% |
| enamel (مینا) | 94.87% | 98.67% | 97.78% | 96.73% |
| pulp (پالپ) | 87.80% | 96.00% | 94.22% | 91.72% |
| **macro** | **92.69%** | **92.44%** | **94.96%** | **92.34%** |

**دقت کلی (Overall accuracy):** 92.44%

## گردش کار پیشنهادی

```bash
# ۱) آموزش (اختیاری اگر چک‌پوینت از قبل موجود است)
python train.py

# ۲) پیش‌بینی دسته‌ای هر پوشه تست
python predict_dentin_test.py
python predict_enamel_test.py
python predict_pulp_test.py

# ۳) ساخت گزارش معیارها
python evaluate_test_predictions.py

# ۴) اختیاری: بررسی یک رادیوگراف کامل
python detect_simple.py
```

## نکات

- دستگاه به‌صورت خودکار انتخاب می‌شود: CUDA → MPS → CPU
- اسکریپت‌های پیش‌بینی تصاویر `.png` / `.jpg` / `.jpeg` را می‌پذیرند
- اگر چک‌پوینت مدل را عوض کردید، قبل از ارزیابی دوباره پیش‌بینی‌ها را اجرا کنید

## مجوز

MIT

---

نسخه انگلیسی: [README.md](README.md)
