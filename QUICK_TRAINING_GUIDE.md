# 🚀 מדריך מהיר לאימון עם Validation מתוקן

## ✅ **הפקודות המעודכנות:**

### **1️⃣ בדיקה מהירה (5 דקות):**
```bash
python phase2_siamese.py \
    --split_strategy realistic_test \
    --subset_size 1000 \
    --epochs 3 \
    --batch_size 32 \
    --data_dir src/data/ \
    --pair_sampling_ratio 0.5
```

### **2️⃣ אימון בינוני (15 דקות):**
```bash
python phase2_siamese.py \
    --split_strategy realistic_test \
    --subset_size 2000 \
    --epochs 8 \
    --batch_size 64 \
    --data_dir src/data/ \
    --pair_sampling_ratio 0.5
```

### **3️⃣ אימון מלא (45 דקות):**
```bash
python phase2_siamese.py \
    --split_strategy realistic_test \
    --subset_size 5000 \
    --epochs 15 \
    --batch_size 128 \
    --data_dir src/data/ \
    --pair_sampling_ratio 0.5 \
    --learning_rate 0.001 \
    --patience 10
```

## 📊 **השוואה - הבדל דרמטי:**

### **A. Validation ישן (שגוי):**
```bash
python phase2_siamese.py \
    --split_strategy random \
    --subset_size 1000 \
    --epochs 5 \
    --data_dir src/data/
# תוצאה: correlation ~0.85 (מטעה!)
```

### **B. Validation חדש (אמיתי!):**
```bash
python phase2_siamese.py \
    --split_strategy realistic_test \
    --subset_size 1000 \
    --epochs 5 \
    --data_dir src/data/
# תוצאה: correlation ~0.70-0.75 (אמיתי!)
```

## 🎯 **מה לצפות:**

### **בפלט תראי:**
```
🎯 Using realistic_test splitting strategy
🎯 Realistic test: 40 proteins, 300 RNAs in validation
✅ 100% validation pairs have both sequences new (like real test!)
✅ Split complete: 188000 train, 12000 validation
📊 Split ratio: 94.0% train, 6.0% validation
```

### **תוצאות אימון:**
```
Epoch 1/5 - Train Loss: 0.135, Val Loss: 0.118, Val Corr: 0.629
Epoch 2/5 - Train Loss: 0.098, Val Loss: 0.102, Val Corr: 0.684
Epoch 3/5 - Train Loss: 0.087, Val Loss: 0.095, Val Corr: 0.712
Epoch 4/5 - Train Loss: 0.079, Val Loss: 0.089, Val Corr: 0.738
Epoch 5/5 - Train Loss: 0.074, Val Loss: 0.085, Val Corr: 0.751
```

**💡 הציונים נמוכים יותר מבעבר - וזה טוב! זה validation אמיתי!**

## 🚀 **עם GPU:**
```bash
python phase2_siamese.py \
    --split_strategy realistic_test \
    --subset_size 10000 \
    --epochs 20 \
    --batch_size 256 \
    --data_dir src/data/ \
    --learning_rate 0.001 \
    --patience 15
```

## 🔧 **פתרון בעיות:**

### **אם מקבלת שגיאה:**
```
'Unknown strategy: realistic_test'
```
**פתרון:** וודאי שעדכנת את `src/data/strategic_split.py`

### **אם validation קטן:**
```
Split complete: 198000 train, 2000 validation
```
**זה בסדר!** Validation קטן אבל קשה יותר = אמיתי יותר

## ✨ **הצעדים הבאים:**

1. **בדוק שעובד:** הרץ בדיקה מהירה
2. **השווה:** הרץ גם עם `random` לראות הבדל
3. **אמן מודל טוב:** הרץ אימון מלא
4. **השתמש לprediction:** עם `FlexiblePredictor`

**עכשיו יש לך validation אמיתי שמכין אותך לטסט של הפרופסור!** 🎯🧬
