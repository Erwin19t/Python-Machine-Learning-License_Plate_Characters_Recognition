def get_characters(results):  
    alphabet = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z','-','0','1','2','3','4','5','6','7','8','9']
    plate = ''
    for i in range(len(results)):    
        plate = plate + alphabet[int(results[i][5])]
    return plate