
![Logo](https://i.ibb.co/DzZMY89/embot.jpg)


# Emotion powered chatbot 🤖

This emotion-powered chatbot uses Streamlit for deployment and OpenCV for an emotion detector (as well as gender and age detection)to generate emotion influenced responses to help the user accordingly.




## Run Locally

#### Clone this Git Repository:

```bash
git clone https://github.com/aroproduction/emotion_powered_chatbot
```

#### Change the working directory to the cloned directory:
```bash
cd emotion_powered_chatbot
```

The customly-trained model trained on "fer2013.csv" dataset (https://www.kaggle.com/datasets/deadskull7/fer2013) with around 63.41% emotion detection accuracy is given as "pr_model.h5".

#### Run 
```bash
pip install -r requirements.txt
```
to install the dependencies.

#### Finally to run the main page:
```bash
streamlit run main.py
```



## Authors

- [@aroproduction](https://www.github.com/aroproduction)
- [@AnubhabMukherjee2003](https://github.com/AnubhabMukherjee2003)
- [@Sharnabho](https://github.com/Sharnabho)


## Screenshots

### Home Page

![App Screenshot](https://i.ibb.co/7RSGVbx/Screenshot-25-3-2024-12284-localhost.jpg)

### Detection Page

![App Screenshot](https://i.ibb.co/SRnKCR6/main-Streamlit-Brave-2-4-2024-1-44-43-PM.png)

### Live Chat based on detection

![App Screenshot](https://i.ibb.co/48bYkx5/Screenshot-25-3-2024-122019-localhost.jpg)



## License

Licensed @ Byte Busters(2024)

