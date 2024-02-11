
![Logo](https://i.ibb.co/DzZMY89/embot.jpg)


# Emotion powered chatbot 🤖

This emotion-powered chatbot uses Streamlit for deployment and OpenCV for an emotion detector (as well as gender and age detection)to generate emotion influenced responses to help the user accordingly.




## Run Locally

To deploy this project first you have to prepare the dataset needed for emotions detctions.

Download "fer2013.csv" from this link:
https://www.kaggle.com/datasets/deadskull7/fer2013

then run 
```bash
python dataset_prepare.py
```

A folder with all the training and testing images will be generated.

the pre-trained model with around 63.41% emotion detection accuracy is given as "pr_model.h5".

Run 
```bash
pip install -r requirements.txt
```
to install the dependencies.

Finally to deploy the project:
```bash
streamlit run main.py
```



## Authors

- [@aroproduction](https://www.github.com/aroproduction)
- [@AnubhabMukherjee2003](https://github.com/AnubhabMukherjee2003)
- [@Sharnabho](https://github.com/Sharnabho)


## Screenshots

![App Screenshot](https://i.ibb.co/SRnKCR6/main-Streamlit-Brave-2-4-2024-1-44-43-PM.png)


