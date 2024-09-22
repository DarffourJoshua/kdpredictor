def GFR(sc, gender, age):
    creatinine = sc * 88.4
    result = 0
    
    if creatinine < 62 and gender == 'female':
        return result = 144 * (creatinine / 61.6)**-0.329 * (0.993)**age
        
    elif creatinine > 62 and gender == 'female':
        return result = 144 * (creatinine / 61.6)**-1.209 * (0.993)**age
        
    elif creatinine < 80 and gender == 'female':
        return result = 141 * (creatinine / 79.2)**-0.411 * (0.993)**age
        
    else
        return result = 141 * (creatinine / 79.2)**-1.209 * (0.993)**age
    
export GFR