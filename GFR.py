def GFR(sc, gender, age):
    creatinine = float(sc) * 88.4
    
    
    if gender == 'Female':
        if creatinine < 62:
            return 144 * (creatinine / 61.6) ** -0.329 * (0.993) ** float(age)
        
        else:
            return 144 * (creatinine / 61.6) ** -1.209 * (0.993) ** float(age)
        
    elif gender == 'Male':
        if creatinine < 80:
            return 141 * (creatinine / 79.2) ** -0.411 * (0.993) ** float(age)
        
        else:
            return 141 * (creatinine / 79.2) ** -1.209 * (0.993) ** float(age)
    
