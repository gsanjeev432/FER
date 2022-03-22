import numpy as np
import json
import cv2
import keras
from bs4 import BeautifulSoup as SOUP 
import re 
import requests as HTTP 

jsonFileName='expression.json'
model=keras.models.model_from_json(json.load(open(jsonFileName,'r')))
model.load_weights('expression.hdf5')
print("done loading weights of trained model")
labell = ["Sad", "Disgust", "Happy", "Neutral", "Sad", "Surprise"]


def main(emotion): 

	# IMDb Url for Drama genre of 
	# movie against emotion Sad 
	if(emotion == "Sad"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=drama&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_7, asc'

	# IMDb Url for Musical genre of 
	# movie against emotion Disgust 
	elif(emotion == "Disgust"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=musical&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_14, asc'

	# IMDb Url for Family genre of 
	# movie against emotion Anger 
	elif(emotion == "Anger"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=family&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_8, asc'

	# IMDb Url for Sport genre of 
	# movie against emotion Fear 
	elif(emotion == "Fear"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=sport&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_18, asc'

	# IMDb Url for Thriller genre of 
	# movie against emotion Enjoyment 
	elif(emotion == "Happy"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=thriller&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_19, asc'

	# IMDb Url for Film_noir genre of 
	# movie against emotion Surprise 
	elif(emotion == "Surprise"): 
		urlhere = 'https://www.imdb.com/search/title/?genres=film_noir&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=B3VQXZZZCMWNCK59ZZW5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_10, asc'

	# HTTP request to get the data of 
	# the whole page 
	response = HTTP.get(urlhere) 
	data = response.text 

	# Parsing the data using 
	# BeautifulSoup 
	soup = SOUP(data, "lxml") 

	# Extract movie titles from the 
	# data using regex 
	title = soup.find_all("a", attrs = {"href" : re.compile(r'\/title\/tt+\d*\/')}) 
	return title 

# Driver Function 
if __name__ == '__main__':
    img=cv2.imread('ang3.jpg')
    cv2.imshow('Input Image',img)
    cropped = cv2.resize(img,(200,200))
    cropped= cropped/255
    cropped = np.reshape(cropped,[1,200,200,3])
    
    classes = model.predict(cropped)
    #print(classes)
    out_lable=np.argmax(classes)
    emotion=out_lable
    print(out_lable)
    out_char = labell[out_lable]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,out_char,(20,20), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('output',img)
    
    emotion = out_char
    a = main(emotion)
    count = 0
    if(emotion == "Anger" or emotion == "Disgust"
						or emotion=="Happy" or emotion=="Neutral" or emotion=="Sad" or emotion=="Surprise" ): 
        for i in a:
            tmp = str(i).split('>')
            if(len(tmp) == 3):
                print(tmp[1][:-3])
            if(count > 13):
                break
            count += 1

cv2.waitKey(0)
cv2.destroyAllWindows()










