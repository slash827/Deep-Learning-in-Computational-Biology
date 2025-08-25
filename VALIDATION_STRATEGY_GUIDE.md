# 🎯 מדריך אסטרטגיות Validation למודל Siamese

## 🔍 **הבעיה הנוכחית:**

הvalidation הנוכחי משתמש ב-**random split** שלא מבחן generalization אמיתי:
```python
# בעייתי:
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```

**בעיות:**
- 🎲 אותם proteins ו-RNAs בשני הסטים
- 🔄 data leakage 
- 📊 לא מבחן יכולת הכללה לרצפים חדשים

## 🎯 **המציאות של הטסט:**

על פי הניתוח של קבצי הטסט:
- **43 חלבונים יחודיים** (test_RBPs2.txt)
- **~120,000 רצפי RNA** (test_seqs.txt) 
- **משימה**: כל חלבון × כל RNA = **5.2M predictions**
- **פלט**: 43 קבצים (אחד לכל חלבון)

## ✅ **אסטרטגיות Validation החדשות:**

### **🏆 1. `test_simulation` (הכי מומלץ!)**
מדמה את תרחיש הטסט האמיתי:
```bash
python phase2_siamese.py --split_strategy test_simulation --subset_size 2000 --epochs 10
```

**מה זה עושה:**
- 🎯 שומר ~43 חלבונים לvalidation (כמו בטסט)
- 🧬 שומר 30% מהRNAs לvalidation
- ✅ מבחן אמיתי לgeneralization

**תוצאות צפויות:**
```
🎯 Test simulation: 10 proteins, 300 RNAs in validation
💡 GENERALIZATION TEST:
   🔮 Novel proteins in val: 10/10 (100.0%)
   🔮 Novel RNAs in val: 300/300 (100.0%)
```

### **🎯 2. `mixed` - מבחן כפול**
```bash
python phase2_siamese.py --split_strategy mixed --subset_size 2000 --epochs 10
```
- ✅ חלק מהproteins חדשים
- ✅ חלק מהRNAs חדשים
- 📊 מבחן generalization מאוזן

### **🧬 3. `protein` - מבחן חלבונים חדשים**
```bash
python phase2_siamese.py --split_strategy protein --subset_size 2000 --epochs 10
```
- 🧬 **רק חלבונים חדשים** בvalidation
- 📊 מבחן יכולת הכללה לproteins

### **🧬 4. `rna` - מבחן RNAs חדשים**
```bash
python phase2_siamese.py --split_strategy rna --subset_size 2000 --epochs 10
```
- 🧬 **רק RNAs חדשים** בvalidation
- 📊 מבחן יכולת הכללה לRNAs

### **🎲 5. `random` - השיטה הנוכחית (לא מומלץ)**
```bash
python phase2_siamese.py --split_strategy random --subset_size 2000 --epochs 10
```
- ⚠️ עלול להטעות לגבי ביצועים
- 🔄 data leakage אפשרי

## 📊 **השוואת תוצאות:**

### **מה לצפות:**

| אסטרטגיה | Validation Correlation צפוי | מה זה מבחן |
|-----------|----------------------------|------------|
| `random` | 0.85+ | ❌ לא מציאותי |
| `test_simulation` | 0.75-0.82 | ✅ מציאותי ביותר |
| `mixed` | 0.78-0.84 | ✅ generalization כללי |
| `protein` | 0.70-0.80 | 🧬 חלבונים חדשים |
| `rna` | 0.72-0.82 | 🧬 RNAs חדשים |

**הציון הנמוך יותר ב-`test_simulation` הוא טוב** - זה אומר שהמודל עובר מבחן אמיתי!

## 🚀 **המלצות לשימוש:**

### **לפיתוח והשוואה:**
```bash
# ראשית - בדוק את כל האסטרטגיות:
python test_split_strategies.py

# אמן עם סימולציה של הטסט (הכי חשוב):
python phase2_siamese.py --split_strategy test_simulation --subset_size 2000 --epochs 10 --batch_size 64

# השווה עם השיטה הנוכחית:
python phase2_siamese.py --split_strategy random --subset_size 2000 --epochs 10 --batch_size 64
```

### **לאימון הסופי:**
```bash
# גרסה מהירה לבדיקה:
python phase2_siamese.py --split_strategy test_simulation --subset_size 1000 --epochs 5

# גרסה מלאה לתוצאות טובות:
python phase2_siamese.py --split_strategy test_simulation --subset_size 5000 --epochs 15 --batch_size 128
```

## 💡 **מה ההבדל יספר לך:**

### **אם random >> test_simulation:**
- המודל "רומה" על נתונים דומים
- הוא לא באמת מכליל טוב
- צריך לשפר architecture או regularization

### **אם test_simulation קרוב ל-random:**
- המודל מכליל מצוין! 🎉
- הוא יעבוד טוב על נתונים חדשים
- אפשר לסמוך על הביצועים

## 🎯 **סיכום:**

**השתמשי ב-`test_simulation` כdefault** - זה הכי קרוב למציאות של הטסט עם 43 חלבונים ו-120K RNAs.

הציונים יהיו יותר נמוכים, אבל הם יהיו **אמיתיים** ויאשרו שהמודל באמת יכול לטפל ברצפים חדשים!
