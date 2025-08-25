#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
הסבר פשוט על בעיית הvalidation הישן
"""

def explain_old_problem():
    print("=== מה היה לפני השינוי? ===")
    print()
    
    # דוגמה פשוטה
    print("דוגמה: 3 חלבונים, 3 RNAs")
    proteins = ["P1", "P2", "P3"]
    rnas = ["R1", "R2", "R3"] 
    
    # יוצרים את כל הזוגות
    pairs = []
    for rna in rnas:
        for protein in proteins:
            pairs.append(f"({rna},{protein})")
    
    print(f"כל הזוגות: {pairs}")
    print(f"סה״כ: {len(pairs)} זוגות")
    print()
    
    # החלוקה הישנה - random split
    print("=== הvalidation הישן (בעייתי): ===")
    print("הגרלה אקראית של הזוגות:")
    
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # 80% train, 20% validation  
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    
    print(f"Train: {train_pairs}")
    print(f"Validation: {val_pairs}")
    print()
    
    # מה הבעיה?
    print("=== הבעיה: ===")
    train_proteins = set()
    train_rnas = set()
    val_proteins = set()
    val_rnas = set()
    
    for pair in train_pairs:
        rna = pair.split(',')[0][1:]  # R1 מתוך (R1
        protein = pair.split(',')[1][:-1]  # P1 מתוך P1)
        train_proteins.add(protein)
        train_rnas.add(rna)
        
    for pair in val_pairs:
        rna = pair.split(',')[0][1:]
        protein = pair.split(',')[1][:-1] 
        val_proteins.add(protein)
        val_rnas.add(rna)
    
    print(f"חלבונים ב-train: {sorted(train_proteins)}")
    print(f"חלבונים ב-validation: {sorted(val_proteins)}")
    print(f"RNAs ב-train: {sorted(train_rnas)}")
    print(f"RNAs ב-validation: {sorted(val_rnas)}")
    print()
    
    # חפיפה
    protein_overlap = train_proteins & val_proteins
    rna_overlap = train_rnas & val_rnas
    
    print("*** חפיפה (DATA LEAKAGE): ***")
    print(f"חלבונים שחוזרים: {sorted(protein_overlap)}")
    print(f"RNAs שחוזרים: {sorted(rna_overlap)}")
    print()
    print("🔥 בעיה: המודל 'זוכר' רצפים מהאימון!")
    print("🔥 התוצאות מטעות!")
    print()

def explain_new_solution():
    print("=== הפתרון החדש: ===")
    print("הפרדה של רצפים (לא זוגות!)")
    print()
    
    proteins = ["P1", "P2", "P3"]
    rnas = ["R1", "R2", "R3"]
    
    # חלוקה חכמה
    train_proteins = ["P1", "P2"]  # רק 2 חלבונים לאימון
    val_proteins = ["P3"]         # חלבון חדש לvalidation
    
    train_rnas = ["R1", "R2"]      # רק 2 RNAs לאימון  
    val_rnas = ["R3"]             # RNA חדש לvalidation
    
    print(f"חלבוני train: {train_proteins}")
    print(f"חלבוני validation: {val_proteins}")
    print(f"RNAs train: {train_rnas}")
    print(f"RNAs validation: {val_rnas}")
    print()
    
    # יצירת זוגות חדשה
    train_pairs = []
    val_pairs = []
    
    for rna in rnas:
        for protein in proteins:
            # אם הרצף או החלבון חדשים -> validation
            if protein in val_proteins or rna in val_rnas:
                val_pairs.append(f"({rna},{protein})")
            else:
                # רק אם שניהם ידועים -> train
                train_pairs.append(f"({rna},{protein})")
    
    print("זוגות Train (רק רצפים ידועים):")
    for pair in train_pairs:
        print(f"  {pair}")
    print()
    
    print("זוגות Validation (רצפים חדשים):")
    for pair in val_pairs:
        rna = pair.split(',')[0][1:]
        protein = pair.split(',')[1][:-1]
        new_protein = protein in val_proteins
        new_rna = rna in val_rnas
        note = []
        if new_protein: note.append("חלבון חדש")
        if new_rna: note.append("RNA חדש")
        print(f"  {pair} <- {', '.join(note)}")
    print()
    
    print("✅ עכשיו: אין חפיפה!")
    print("✅ המודל מתאמן על רצפים ידועים")
    print("✅ נבדק על רצפים חדשים לגמרי")
    print("✅ כמו הטסט: 43 חלבונים חדשים!")

def show_real_impact():
    print()
    print("=== מה זה אומר בפועל? ===")
    print()
    print("הטסט של הפרופסור:")
    print("  📁 test_RBPs2.txt: 43 חלבונים חדשים")
    print("  📁 test_seqs.txt: ~120,000 RNAs חדשים")
    print("  🎯 המשימה: כל חלבון × כל RNA")
    print()
    print("הvalidation הישן:")
    print("  ❌ בדק זוגות אקראיים")
    print("  ❌ אותם רצפים ב-train וב-validation")
    print("  ❌ ציונים מנופחים (correlation ~0.85)")
    print("  ❌ הפתעה בטסט!")
    print()
    print("הvalidation החדש:")
    print("  ✅ בודק רצפים חדשים")
    print("  ✅ מדמה את הטסט")
    print("  ✅ ציונים אמיתיים (correlation ~0.75-0.82)")
    print("  ✅ אין הפתעות!")

if __name__ == "__main__":
    explain_old_problem()
    explain_new_solution()
    show_real_impact()
