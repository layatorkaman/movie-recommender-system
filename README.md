# سیستم پیشنهاددهنده فیلم (Movie Recommender System)

این پروژه با زبان Python و با هدف پیاده‌سازی و مقایسه روش‌های مختلف سیستم‌های پیشنهاددهنده (Recommender Systems) توسعه داده شده است.

در این پروژه سه رویکرد پیاده‌سازی و ارزیابی شده‌اند:

- Baseline Recommender
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering

## ویژگی‌های پروژه

- بارگذاری و اعتبارسنجی داده‌های MovieLens
- تحلیل اکتشافی داده‌ها (EDA)
- ساخت ماتریس User-Item
- پیاده‌سازی روش‌های User-Based و Item-Based Collaborative Filtering
- محاسبه شباهت با استفاده از Pearson Correlation
- بهینه‌سازی محاسبات Item-Based با استفاده از عملیات برداری NumPy
- ارزیابی مدل‌ها با معیارهای MAE و RMSE
- مقایسه Prediction Coverage و زمان اجرا
- تست خودکار با استفاده از pytest

## مجموعه داده (Dataset)

در این پروژه از مجموعه داده [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) استفاده شده است.

مشخصات مجموعه داده:

| ویژگی | مقدار |
|---|---:|
| تعداد امتیازها (Ratings) | 100,836 |
| تعداد کاربران (Users) | 610 |
| تعداد فیلم‌ها (Movies) | 9,724 |
| میانگین امتیازها | 3.50 |
| بازه امتیازدهی | 0.5 تا 5 |

فایل‌های `ratings.csv` و `movies.csv` باید در مسیر زیر قرار بگیرند:

```text
data/raw/ml-latest-small/
```

## روش‌های پیشنهاددهی

### ۱. Baseline Recommender

در این روش، امتیاز پیش‌بینی‌شده برای هر فیلم بر اساس میانگین امتیازهای ثبت‌شده برای همان فیلم محاسبه می‌شود. اگر فیلمی در داده‌های آموزشی وجود نداشته باشد، میانگین کلی امتیازها به‌عنوان مقدار جایگزین استفاده می‌شود.

### ۲. User-Based Collaborative Filtering

در این روش، کاربران بر اساس الگوی امتیازدهی به فیلم‌های مشترک با یکدیگر مقایسه می‌شوند. سپس با استفاده از میزان شباهت کاربران و امتیازهای آن‌ها، امتیاز فیلم‌های مشاهده‌نشده پیش‌بینی می‌شود.

### ۳. Item-Based Collaborative Filtering

در این روش، شباهت میان فیلم‌ها بر اساس امتیازهایی که کاربران مشترک به آن‌ها داده‌اند محاسبه می‌شود. برای افزایش سرعت محاسبات، در بخش پیش‌بینی از عملیات برداری NumPy استفاده شده است.

## ارزیابی و مقایسه مدل‌ها

مدل‌ها با استفاده از روش تقسیم داده‌های آموزش و آزمون (Train-Test Split) و نسبت ۸۰ به ۲۰ ارزیابی شده‌اند.

در آزمایش مقایسه‌ای، ۵۰۰ نمونه از داده‌های آزمون انتخاب شده‌اند. برای مقایسه منصفانه دو مدل Collaborative Filtering، معیارهای خطا روی ۴۵۲ نمونه‌ای محاسبه شده‌اند که هر دو مدل برای آن‌ها پیش‌بینی تولید کرده‌اند.

### نتایج آزمایش

| مدل | MAE | RMSE | Prediction Coverage |
|---|---:|---:|---:|
| Baseline | 0.7738 | 0.9729 | 100% |
| User-Based | **0.6829** | **0.8769** | 95.60% |
| Item-Based | 0.7215 | 0.9204 | 90.40% |

### زمان اجرا

| مدل | زمان اجرا |
|---|---:|
| User-Based | 47.48 ثانیه |
| Item-Based | 6.47 ثانیه |

در این آزمایش، مدل User-Based کمترین خطای پیش‌بینی را داشته است؛ در مقابل، مدل Item-Based تقریباً ۷.۳ برابر سریع‌تر اجرا شده است.

توجه: زمان اجرا به سخت‌افزار و شرایط اجرای برنامه وابسته است و ممکن است در سیستم‌های مختلف تغییر کند.

## نصب و راه‌اندازی

ابتدا یک محیط مجازی ایجاد کنید:

```cmd
python -m venv venv
```

محیط مجازی را در Windows فعال کنید:

```cmd
venv\Scripts\activate
```

سپس وابستگی‌های پروژه را نصب کنید:

```cmd
pip install -r requirements.txt
```

## اجرای پروژه

اجرای تحلیل اکتشافی داده‌ها:

```cmd
python run_exploration.py
```

اجرای مدل User-Based:

```cmd
python run_recommender.py
```

اجرای مدل Item-Based:

```cmd
python run_item_based.py
```

اجرای ارزیابی و مقایسه مدل‌ها:

```cmd
python run_evaluation.py
```

## اجرای تست‌ها

برای اجرای تست‌های پروژه از دستور زیر استفاده کنید:

```cmd
pytest -v
```

در آخرین اجرای تست‌ها، هر ۳۹ تست با موفقیت اجرا شده‌اند.

## ساختار پروژه

```text
Recommender system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   └── model_comparison.md
│
├── src/
│   ├── data/
│   │   └── loader.py
│   │
│   ├── evaluation/
│   │   └── evaluation.py
│   │
│   ├── exploration/
│   │   ├── analysis.py
│   │   ├── check_matrix.py
│   │   └── check_similarity.py
│   │
│   └── recommenders/
│       ├── baseline.py
│       ├── user_based.py
│       └── item_based.py
│
├── tests/
│   ├── test_evaluation.py
│   ├── test_user_based.py
│   └── test_item_based.py
│
├── run_exploration.py
├── run_recommender.py
├── run_item_based.py
├── run_evaluation.py
├── requirements.txt
└── README.md
```

## محدودیت‌ها

- آزمایش مقایسه‌ای فعلی بر اساس ۵۰۰ نمونه از داده‌های آزمون انجام شده است.
- نتایج به نحوه تقسیم داده‌ها و پارامترهای انتخاب‌شده وابسته هستند.
- مدل‌های Collaborative Filtering ممکن است برای تمام ترکیب‌های کاربر و فیلم پیش‌بینی تولید نکنند.
- زمان اجرای گزارش‌شده مختص محیط آزمایش فعلی است.

## توسعه‌های آینده

- تنظیم و بهینه‌سازی پارامترهای مدل‌ها (Hyperparameter Tuning)
- پیاده‌سازی الگوریتم‌های پیشنهاددهی بیشتر
- ارزیابی پیشنهادهای Top-N با معیارهایی مانند Precision@K و Recall@K
- بهبود عملکرد در مواجهه با داده‌های پراکنده (Sparse Data)

## فناوری‌های استفاده‌شده

- Python
- Pandas
- NumPy
- Scikit-learn
- Pytest
- Git