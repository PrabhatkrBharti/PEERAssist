import re

replaceWords = [[" et al. "," and others "] , [" i.e. "," that is "] , [" vs. "," versus "], [" e.g. "," example given "], [" eg. "," example given "], [" n.b. "," note well "], [" approx. "," approximately "], [" dept. "," department "], [" no. "," number "], [" misc. "," miscellaneous "], [" min. "," minute "]]

def scanWhitSpaces(txt):
	result = ""
	flag = True
	for ch in txt:
		if ch==" ":
			if flag==False:
				flag = True
				result += ch
		else:
			result += ch
			flag = False
	return result
	
def scanDots(txt):
	result = txt.replace(". .","..")
	result = result.replace(". .","..")
	
	result = result.replace("....","..")
	result = result.replace("...","..")
	result = result.replace("....","..")
	result = result.replace("...","..")
	
	result = result.replace(".."," continuation ")
	return result
	
def scanAlphaNum(txt):
	return re.sub('[^A-Za-z0-9 ]+', ' ', txt)

def sentence_tokenize(paragraph):
	result = []
	if paragraph==None:
		return result
	if len(paragraph)==0:
		return result
		
	paragraph = paragraph.lower()
	paragraph = scanDots(paragraph)
	for dummy in replaceWords:
		paragraph = paragraph.replace(dummy[0] , dummy[1])
	paragraph = paragraph.split(". ")
	for smallPara in paragraph:
		sentences = smallPara.split("\n")
		for sentence in sentences:
			if len(sentence)>0:
				dummy = scanWhitSpaces(scanAlphaNum(sentence))
				if len(dummy)>0:
					result.append([dummy])
	return result

